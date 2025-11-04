import os
import base64
import requests
from io import BytesIO

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None


# Get OpenAI API Key from environment variable
api_key = os.environ.get('OPENAI_API_KEY', '')

# Get Gemini API Key from environment variable
gemini_api_key = os.environ.get('GEMINI_API_KEY', '')

# TODO(kuanfang): Maybe also support free form-responses.
headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {api_key}'
}

DEFAULT_LLM_MODEL_NAME = 'gpt-4'
DEFAULT_VLM_MODEL_NAME = 'gpt-4o'


def encode_image_from_file(image_path):
    # Function to encode the image
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def encode_image_from_pil(image):
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def prepare_inputs(messages,
                   images,
                   meta_prompt,
                   model_name,
                   local_image):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


def request_gpt(message,
                images,
                meta_prompt='',
                model_name=None,
                local_image=False):

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs(message,
                             images,
                             meta_prompt=meta_prompt,
                             model_name=model_name,
                             local_image=local_image)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print('\n' + '='*70)
        print('ERROR: Invalid API response')
        print('='*70)
        print(f'Status Code: {response.status_code}')
        print(f'Response: {response.json()}')
        print('='*70)
        raise Exception(f"Failed to get valid response from OpenAI API: {response.json()}")

    return res


def prepare_inputs_incontext(
        messages,
        images,
        meta_prompt,
        model_name,
        local_image,
        example_images,
        example_responses,
):

    user_content = []

    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        content = {
            'type': 'text',
            'text': message,
        }
        user_content.append(content)

    if not isinstance(images, list):
        images = [images]

    for example_image, example_response in zip(
            example_images, example_responses):
        if local_image:
            base64_image = encode_image_from_file(example_image)
        else:
            base64_image = encode_image_from_pil(example_image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

        content = {
            'type': 'text',
            'text': example_response,
        }
        user_content.append(content)

    for image in images:
        if local_image:
            base64_image = encode_image_from_file(image)
        else:
            base64_image = encode_image_from_pil(image)

        content = {
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        }
        user_content.append(content)

    payload = {
        'model': model_name,
        'messages': [
            {
                'role': 'system',
                'content': [
                    meta_prompt
                ]
            },
            {
                'role': 'user',
                'content': user_content,
            }
        ],
        'max_tokens': 800
    }

    return payload


def request_gpt_incontext(
        message,
        images,
        meta_prompt='',
        example_images=None,
        example_responses=None,
        model_name=None,
        local_image=False):

    # TODO(kuan): A better interface should allow interleaving images and
    # messages in the inputs.

    if model_name is None:
        if images is [] or images is None:
            model_name = DEFAULT_LLM_MODEL_NAME
        else:
            model_name = DEFAULT_VLM_MODEL_NAME

    payload = prepare_inputs_incontext(
        message,
        images,
        meta_prompt=meta_prompt,
        model_name=model_name,
        local_image=local_image,
        example_images=example_images,
        example_responses=example_responses)

    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers=headers,
        json=payload)

    try:
        res = response.json()['choices'][0]['message']['content']
    except Exception as e:
        print('\n' + '='*70)
        print('ERROR: Invalid API response')
        print('='*70)
        print(f'Status Code: {response.status_code}')
        print(f'Response: {response.json()}')
        print('='*70)
        raise Exception(f"Failed to get valid response from OpenAI API: {response.json()}")

    return res

# ============================================================================
# GEMINI API SUPPORT
# ============================================================================

DEFAULT_GEMINI_MODEL_NAME = 'gemini-2.5-pro'


def request_gemini(message,
                   images,
                   meta_prompt='',
                   model_name=None):
    """
    Request Gemini API (similar interface to request_gpt).

    Args:
        message: Text prompt or list of text prompts
        images: PIL Image or list of PIL Images
        meta_prompt: System prompt (prepended to message)
        model_name: Gemini model name (default: gemini-2.5-pro)

    Returns:
        Response text from Gemini
    """
    if not GEMINI_AVAILABLE:
        raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    # Configure Gemini API
    genai.configure(api_key=gemini_api_key)

    # Set default model
    if model_name is None:
        model_name = DEFAULT_GEMINI_MODEL_NAME

    # Create model
    model = genai.GenerativeModel(model_name)

    # Prepare content
    content = []

    # Add system prompt as first message if provided
    if meta_prompt:
        content.append(meta_prompt + "\n\n")

    # Add messages
    if not isinstance(message, list):
        message = [message]
    for msg in message:
        content.append(msg)

    # Add images
    if images is not None:
        if not isinstance(images, list):
            images = [images]
        for img in images:
            content.append(img)

    try:
        # Generate content
        response = model.generate_content(content)
        res = response.text
        return res
    except Exception as e:
        print('\n' + '='*70)
        print('ERROR: Invalid Gemini API response')
        print('='*70)
        print(f'Error: {e}')
        print('='*70)
        raise Exception(f"Failed to get valid response from Gemini API: {e}")
