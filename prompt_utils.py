def create_rag_prompt(message, pdf_name, results):
    filename = pdf_name.split('\\')[-1] if '\\' in pdf_name else pdf_name.split('/')[-1]
    system_content = (
        "Please provide an answer based on your general knowledge.\n"
        "Please use the language in your response that matches the language in which the question is asked.\n"
        "When answering, please use a polite tone and answer systematically and with good visibility.\n"
        "I want you to act as a document analyzer. I will provide you with a list of documents that need to be analyzed and analyzed. \
        You will also provide me with an analysis of each document. My first request is I need help analyzing a document ."
    )
    context = f"Current fileName: {filename}" "\n" f"fileData: {results}"
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": context},
        {"role": "user", "content": message}
    ]

def create_image_prompt(message, encoded_image):
    system_content = (
        "Please provide an answer based on your general knowledge.\n"
        "Please use the language in your response that matches the language in which the question is asked.\n"
        "When answering, please use a polite tone and answer systematically and with good visibility."
        "I want you to act as an image analysis tool. I will provide you with an image and you will analyze it using image analysis tools. \
        My first request is I need an image information."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": message},
        {"role": "user", "content": f"data:image/jpeg;base64,{encoded_image}"}
    ]

def create_default_prompt(message):
    system_content = (
        "Please provide an answer based on your general knowledge.\n"
        "Please use the language in your response that matches the language in which the question is asked.\n"
        "When answering, please use a polite tone and answer systematically and with good visibility."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": message}
    ]