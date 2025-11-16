"""Query Type Classifier - Detects the type of query from available task types."""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

# Hardcoded list of task types found in the dataset
TASK_TYPES = [
    'Action Reasoning',
    'Action Recognition',
    'Attribute Perception',
    'Counting Problem',
    'Information Synopsis',
    'OCR Problems',
    'Object Reasoning',
    'Object Recognition',
    'Spatial Perception',
    'Spatial Reasoning',
    'Temporal Perception',
    'Temporal Reasoning'
]

# Few-shot examples for each task type (NOT from eval_subset.json)
# Format: (question, task_type) pairs to show clear patterns
FEW_SHOT_EXAMPLES = [
    # Attribute Perception vs Object Recognition boundary
    ("What color are the mountains?", "Attribute Perception"),  # property only
    ("Which sport is shown?", "Object Recognition"),             # category/type
    ("What pattern does the cat have?", "Attribute Perception"), # appearance
    
    # Action Recognition vs Action Reasoning boundary  
    ("What step was taken after filling?", "Action Recognition"),     # what happened
    ("Why did they collect the lava safely with a black mask?", "Action Reasoning"),  # reasoning
    ("According to the video, which ingredients are NOT used?", "Action Reasoning"),  # why not
    
    # Object Reasoning vs Information Synopsis boundary (YOUR MAIN ERROR)
    ("What do the blue/green keys mean?", "Object Reasoning"),      # inference about object meaning
    ("What is the video mainly about?", "Information Synopsis"),    # overall theme/topic
    ("Can you infer why the Triangulum Galaxy is with Andromeda?", "Object Reasoning"),  # inference
    ("What is this video primarily about?", "Information Synopsis"), # main topic
    
    # Spatial Reasoning vs Spatial Perception boundary
    ("What is visible in the background?", "Spatial Perception"),     # what/where
    ("Why does the smoke flow towards the lamp?", "Spatial Reasoning"),  # causal/why
    
    # Temporal Reasoning vs Temporal Perception boundary
    ("In which part is the woman interviewed?", "Temporal Perception"),  # when/where in time
    ("What is the sequence of steps?", "Temporal Reasoning"),           # chronological order
    
    # Counting vs OCR boundary
    ("How many streams did the cat cross?", "Counting Problem"),        # count objects
    ("What temperature is shown on the display?", "OCR Problems"),      # read displayed text
]


class QueryTypeClassifier:
    """Classify queries into task types using an LLM."""
    
    def __init__(self, model_name='Qwen/Qwen3-4B-Instruct-2507'):
        """
        Initialize the query type classifier.
        
        Args:
            model_name: HuggingFace model name for classification
        """
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map='auto'
        )
        self.model.eval()
        print("Model loaded successfully")
    
    def classify(self, question: str) -> str:
        """
        Classify a single query/question into one of the available task types.
        
        Args:
            question: The question/query string to classify
            
        Returns:
            Predicted task type string
        """
        # Use ALL examples (36 total) for better coverage, but organize by type
        example_by_type = {}
        for q, t in FEW_SHOT_EXAMPLES:
            if t not in example_by_type:
                example_by_type[t] = []
            example_by_type[t].append((q, t))
        
        # Build examples grouped by type for clarity
        few_shot_examples = "Examples:\n\n"
        for task_type in TASK_TYPES:
            if task_type in example_by_type:
                few_shot_examples += f"{task_type}:\n"
                for example_question, example_type in example_by_type[task_type]:
                    few_shot_examples += f"  Q: {example_question}\n  A: {example_type}\n"
                few_shot_examples += "\n"
        
        # Create prompt with comprehensive few-shot examples
        task_types_str = ', '.join(TASK_TYPES)
        prompt = f"""Classify the following question into one of these task types: {task_types_str}

{few_shot_examples}CLASSIFICATION RULES (choose ONE that best applies):

**Attribute Perception**: Asks about PROPERTIES/APPEARANCE (color, size, texture, pattern, physique)
  ✓ "What color are the mountains?"
  ✗ "Which items are present?" (that's Recognition)
  ✗ "Why is it that color?" (that's Reasoning)

**Object Recognition**: Asks WHAT TYPE/CATEGORY, WHICH ITEM EXISTS
  ✓ "Which sport is shown?"
  ✓ "Which item does not appear?"
  ✗ "What does this mean?" (that's Reasoning)

**Object Reasoning**: Asks WHAT CAN BE INFERRED about objects/their meaning/relationships
  ✓ "What do the blue/green keys mean?"
  ✓ "Can you infer why this galaxy is nearby?"
  ✓ "Which elements are not present in the painting?" (inferring what's absent)
  ✗ "What is the video about?" (that's Synopsis)

**Information Synopsis**: Asks MAIN TOPIC, OVERALL THEME, WHAT THE VIDEO IS PRIMARILY ABOUT
  ✓ "What is the video mainly about?"
  ✓ "What does this video demonstrate?"
  ✗ "What can be inferred from details?" (that's Reasoning)

**Action Recognition**: Asks WHAT ACTION HAPPENED, WHO DID IT, WHAT STEP WAS TAKEN, WHICH ACTION
  ✓ "What was the next step after filling?"
  ✓ "What action was demonstrated?"
  ✗ "Why was this action taken?" (that's Reasoning)

**Action Reasoning**: Asks WHY AN ACTION WAS TAKEN, PURPOSE, REASON
  ✓ "Why did they use this technique?"
  ✓ "What is the purpose of this procedure?"
  ✗ "What step was taken?" (that's Recognition)

**Spatial Perception**: Asks WHERE, WHAT IS VISIBLE IN LOCATION, POSITION RELATIVE TO
  ✓ "What is visible in the background?"
  ✓ "Where is the object positioned?"
  ✗ "Why is it positioned there?" (that's Reasoning)

**Spatial Reasoning**: Asks WHY (spatial), WHAT CAUSES movement or spatial relationship
  ✓ "Why does smoke flow towards the lamp?"
  ✓ "What explains this spatial relationship?"

**Temporal Perception**: Asks WHEN, IN WHICH PART, AT WHAT POINT IN TIME
  ✓ "In which part of the video does this appear?"
  ✓ "When did this event occur?"
  ✗ "In what order?" (that's Reasoning)

**Temporal Reasoning**: Asks WHAT ORDER, CHRONOLOGICAL SEQUENCE, WHAT COMES NEXT
  ✓ "What is the sequence of steps?"
  ✓ "In what order do these occur?"

**Counting Problem**: Asks HOW MANY, COUNT, TOTAL NUMBER
  ✓ "How many streams did the cat cross?"
  ✗ "What number is displayed?" (that's OCR)

**OCR Problems**: Asks WHAT TEXT/NUMBER/TIME IS DISPLAYED (reading visible/written text)
  ✓ "What time is shown on the clock?"
  ✓ "What temperature is displayed?"
  ✗ "How many are there?" (that's Counting)'

Question: {question}

Return ONLY the task type name that best matches the question. Do not include any explanation or additional text.

Task Type:"""
        
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048  # Increased for all 36 examples + distinctions
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        # Extract task type from response
        predicted_type = self._extract_task_type(response)
        return predicted_type
    
    def _extract_task_type(self, response: str) -> str:
        """
        Extract task type from model response, matching against available types.
        
        Args:
            response: Model response string
            
        Returns:
            Matched task type or most likely match as fallback
        """
        response = response.strip().lower()
        
        # Try exact match first (case-insensitive)
        for task_type in TASK_TYPES:
            if task_type.lower() == response:
                return task_type
        
        # Try substring match (task type appears in response)
        for task_type in TASK_TYPES:
            if task_type.lower() in response:
                return task_type
        
        # Try partial word match - check if key words from task types appear
        response_words = set(response.split())
        best_match = None
        best_score = 0
        
        for task_type in TASK_TYPES:
            task_words = set(task_type.lower().split())
            # Count matching words
            match_count = len(task_words.intersection(response_words))
            if match_count > best_score and match_count > 0:
                best_score = match_count
                best_match = task_type
        
        if best_match:
            return best_match
        
        # Final fallback: try to find any word that might indicate a task type
        # Check for keywords that strongly indicate certain types
        keyword_mapping = {
            'action': ['Action Recognition', 'Action Reasoning'],
            'object': ['Object Recognition', 'Object Reasoning'],
            'spatial': ['Spatial Perception', 'Spatial Reasoning'],
            'temporal': ['Temporal Perception', 'Temporal Reasoning'],
            'counting': ['Counting Problem'],
            'count': ['Counting Problem'],
            'ocr': ['OCR Problems'],
            'text': ['OCR Problems'],
            'attribute': ['Attribute Perception'],
            'information': ['Information Synopsis'],
            'synopsis': ['Information Synopsis'],
        }
        
        for keyword, candidates in keyword_mapping.items():
            if keyword in response:
                # If we can distinguish further
                if 'reasoning' in response or 'why' in response or 'reason' in response:
                    if 'Action Reasoning' in candidates:
                        return 'Action Reasoning'
                    if 'Object Reasoning' in candidates:
                        return 'Object Reasoning'
                    if 'Spatial Reasoning' in candidates:
                        return 'Spatial Reasoning'
                    if 'Temporal Reasoning' in candidates:
                        return 'Temporal Reasoning'
                elif 'recognition' in response or 'what type' in response or 'what kind' in response:
                    if 'Action Recognition' in candidates:
                        return 'Action Recognition'
                    if 'Object Recognition' in candidates:
                        return 'Object Recognition'
                elif 'perception' in response or 'where' in response or 'when' in response:
                    if 'Spatial Perception' in candidates:
                        return 'Spatial Perception'
                    if 'Temporal Perception' in candidates:
                        return 'Temporal Perception'
                # Return first candidate if no further distinction
                if candidates:
                    return candidates[0]
        
        # Last resort: don't default to first, try to parse the response better
        print(f"Warning: Could not match '{response}' to any task type. Returning first available type.")
        return TASK_TYPES[0]


def classify_question(question: str, model_name: str = 'Qwen/Qwen3-4B-Instruct-2507') -> str:
    """
    Convenience function to classify a single question.
    
    Args:
        question: The question string to classify
        model_name: Model to use for classification
        
    Returns:
        Predicted task type string
    """
    classifier = QueryTypeClassifier(model_name=model_name)
    return classifier.classify(question)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify a question by task type')
    parser.add_argument('question', type=str, help='Question string to classify')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Model name for classification')
    
    args = parser.parse_args()
    
    classifier = QueryTypeClassifier(model_name=args.model)
    task_type = classifier.classify(args.question)
    
    print(f"\nQuestion: {args.question}")
    print(f"Predicted Task Type: {task_type}")