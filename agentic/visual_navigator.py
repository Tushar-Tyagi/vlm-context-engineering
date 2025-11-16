"""
Visual agent for EKG traversal using Qwen3-VL-2B.
"""

import torch
import numpy as np
from PIL import Image
import re
import cv2
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class VisualEventNavigator:
    """Agent that explores EKG by looking at frames."""
    
    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct"):
        print(f"Loading Visual Navigator: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.current_video_path = None
    
    def navigate(self, initial_event, ekg_data, question, video_path, max_hops=5):
        """Explore EKG starting from initial event."""
        # Extract video frames
        video_frames = self._extract_video_frames(video_path)
        
        # Build lookup structures
        events_by_id = {e['event_id']: e for e in ekg_data['events']}
        relationships = self._build_relationship_map(ekg_data['relationships'])
        
        # Exploration loop
        visited = []
        visited_ids = set()
        current_event_id = initial_event['event_id']
        trace = []
        self.current_trace = trace
        
        print(f"\nStarting navigation from event {current_event_id}")
        
        for hop in range(max_hops + 1):
            current_event = events_by_id[current_event_id]
            visited.append(current_event)
            visited_ids.add(current_event_id)
            
            print(f"\nHop {hop}/{max_hops}: Event {current_event_id}")
            
            # Check available actions
            has_next = (relationships['next'].get(current_event_id) is not None and 
                    relationships['next'][current_event_id] not in visited_ids)
            has_prev = (relationships['prev'].get(current_event_id) is not None and
                    relationships['prev'][current_event_id] not in visited_ids)
            
            # Agent decision
            decision = self._agent_decision(
                current_event, visited, question, video_frames,
                has_next, has_prev, hop, max_hops
            )
            
            trace.append({
                'hop': hop,
                'event_id': current_event_id,
                'action': decision['action'],
                'reasoning': decision['reasoning']
            })
            
            # Execute action
            if decision['action'] == 'answer':
                break
            elif decision['action'] == 'next':
                if has_next:
                    current_event_id = relationships['next'][current_event_id]
                else:
                    break       
            elif decision['action'] == 'previous':
                if has_prev:
                    current_event_id = relationships['prev'][current_event_id]
            else:
                break
        
        return visited, trace
    
    def _agent_decision(self, current_event, visited, question, video_frames,
                    has_next, has_prev, hop, max_hops):
        """Agent looks at frames and decides next action."""
        # Sample frames from current event
        sampled_frames = self._sample_event_frames(current_event, video_frames, n=3)
        
        # Build prompt
        available_actions = []
        if has_next:
            available_actions.append("- NEXT: Navigate to the next event")
        if has_prev:
            available_actions.append("- PREVIOUS: Navigate to the previous event")
        available_actions.append("- ANSWER: Stop exploring, you have enough context")
        
        actions_text = "\n".join(available_actions)
        
        # Build context from trace reasoning instead of descriptions
        if len(visited) == 1:
            context_summary = "This is your first event."
        else:
            # Get reasoning from trace
            trace_context = []
            for i, step in enumerate(self.current_trace):
                trace_context.append(f"Hop {step['hop']}: {step['action'].upper()} - {step['reasoning']}")
            context_summary = "\n".join(trace_context)
        
        prompt = f"""You are exploring a video to answer: "{question}"

    Current event (hop {hop}/{max_hops}):

    Below are frames from this event.

    Your exploration so far:
    {context_summary}

    Available actions:
    {actions_text}

    IMPORTANT: Only choose ANSWER if you are confident you have enough context to answer the question correctly.
    
    Choose your action and explain why in this format:
    ACTION: [NEXT/PREVIOUS/ANSWER]
    REASONING: [One sentence explanation]"""

        # Build message with frames
        content = [{"type": "text", "text": prompt}]
        for frame in sampled_frames:
            pil_frame = Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
            content.append({"type": "image", "image": pil_frame})
        
        messages = [{"role": "user", "content": content}]
        
        # Generate decision
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=128, do_sample=False
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse decision
        return self._parse_decision(response, has_next, has_prev)
    
    def _parse_decision(self, response, has_next, has_prev):
        """Extract action and reasoning from response."""
        print(f"Debug Response: {response}")
        action_match = re.search(r'ACTION:\s*(NEXT|PREVIOUS|ANSWER)', response, re.IGNORECASE)
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        
        if action_match:
            action = action_match.group(1).lower()
            if action == 'next' and not has_next:
                action = 'answer'
            elif action == 'previous' and not has_prev:
                action = 'answer'
        else:
            response_lower = response.lower()
            if 'answer' in response_lower:
                action = 'answer'
            elif 'next' in response_lower and has_next:
                action = 'next'
            elif 'previous' in response_lower and has_prev:
                action = 'previous'
            else:
                action = 'answer'
        
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response[:200]
        return {'action': action, 'reasoning': reasoning}
    
    def _sample_event_frames(self, event, video_frames, n=3):
        """Sample n frames evenly from event."""
        cap = cv2.VideoCapture(str(self.current_video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        cap.release()
        
        frame_indices = event['frame_indices']
        fps_indices = [idx // frame_interval for idx in frame_indices]
        
        if len(fps_indices) <= n:
            sampled_indices = fps_indices
        elif n == 1:
            sampled_indices = [fps_indices[len(fps_indices) // 2]]
        elif n == 2:
            sampled_indices = [fps_indices[0], fps_indices[-1]]
        else:
            step = (len(fps_indices) - 1) / (n - 1)
            sampled_indices = [fps_indices[int(round(i * step))] for i in range(n)]
        
        frames = []
        for idx in sampled_indices:
            if idx < len(video_frames):
                frames.append(video_frames[idx])
        return frames
    
    def _extract_video_frames(self, video_path):
        """Extract all frames at 1 FPS."""
        self.current_video_path = video_path
        
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / 1)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
        
        cap.release()
        return frames
    
    def _build_relationship_map(self, relationships):
        """Build lookup for NEXT/PREV relationships."""
        next_map = {}
        prev_map = {}
        for rel in relationships:
            if rel['type'] == 'NEXT':
                next_map[rel['source']] = rel['target']
                prev_map[rel['target']] = rel['source']
        return {'next': next_map, 'prev': prev_map}