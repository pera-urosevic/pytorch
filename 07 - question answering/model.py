import torch
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model = '~model/distilbert-base-uncased-distilled-squad'
question_answerer = pipeline("question-answering", model=model, device=device)


def load(path):
    with open(path) as f:
        return f.read()


def get_local_context(context, answer_start, answer_end, radius=30):
    prefix = ''
    local_context_start = answer_start - radius
    if (local_context_start < 0):
        local_context_start = 0
    else:
        prefix = '...'
    postfix = ''
    length = len(context)
    local_context_end = answer_end + radius
    if (local_context_end > length):
        local_context_end = length
    else:
        postfix = '...'
    local_context = f"**{context[answer_start:answer_end]}**"
    local_context = f"{context[local_context_start:answer_start]}{local_context}{context[answer_end:local_context_end]}"
    local_context = f"{prefix}{local_context}{postfix}"
    local_context = local_context.replace('\n', ' ')
    return local_context


def ask(context, question):
    result = question_answerer(question=question, context=context)
    answer = result['answer']
    score = round(result['score'], 4)
    start = result['start']
    end = result['end']
    local_context = get_local_context(context, start, end)

    print()
    print(f"#### {question}")
    print(f"*{answer}* [score={score}]")
    print(f"> {local_context}")
    print()
