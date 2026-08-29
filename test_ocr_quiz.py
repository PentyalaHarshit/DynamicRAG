import easyocr
import re
from agents.quiz_agent import solve_quiz_query

reader = easyocr.Reader(['en'], gpu=False)
img_path = r'C:\Users\SAI\.gemini\antigravity-ide\brain\0f21f6f6-6003-4532-983d-f6a594d1faaf\.user_uploaded\media_1787971930991.png'
results = reader.readtext(img_path)

# Extract bounding boxes
boxes = []
for bbox, text, prob in results:
    x_mid = (bbox[0][0] + bbox[1][0]) / 2.0
    y_mid = (bbox[0][1] + bbox[2][1]) / 2.0
    boxes.append({'x': x_mid, 'y': y_mid, 'text': text, 'prob': prob, 'top': bbox[0][1], 'bot': bbox[2][1]})

# Sort by top
boxes.sort(key=lambda b: b['top'])

# Group into lines
lines = []
curr_line = []
curr_y = None
for b in boxes:
    if curr_y is None or abs(b['top'] - curr_y) < 18:
        curr_line.append(b)
        curr_y = b['top'] if curr_y is None else (curr_y + b['top'])/2.0
    else:
        curr_line.sort(key=lambda item: item['x'])
        lines.append({'y': curr_y, 'text': ' '.join(item['text'] for item in curr_line)})
        curr_line = [b]
        curr_y = b['top']
if curr_line:
    curr_line.sort(key=lambda item: item['x'])
    lines.append({'y': curr_y, 'text': ' '.join(item['text'] for item in curr_line)})

# Find where question ends (first question mark)
q_end_idx = -1
for i, l in enumerate(lines):
    if '?' in l['text']:
        q_end_idx = i
        break

if q_end_idx >= 0:
    q_text = ' '.join(lines[k]['text'] for k in range(q_end_idx + 1))
    opt_lines = lines[q_end_idx + 1:]
    
    # Group opt_lines into blocks by gap > 35px
    blocks = []
    curr_block = []
    prev_y = None
    for ol in opt_lines:
        if 'Next' in ol['text'] or 'Feedback' in ol['text']:
            continue
        if prev_y is not None and (ol['y'] - prev_y) > 35:
            if curr_block:
                blocks.append(' '.join(curr_block))
                curr_block = []
        curr_block.append(ol['text'])
        prev_y = ol['y']
    if curr_block:
        blocks.append(' '.join(curr_block))
    
    print('\nDetected Question:')
    print(q_text)
    print('\nDetected Blocks:')
    synth_query = q_text + ' '
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    for idx, b in enumerate(blocks):
        print(f'{letters[idx]}: {b}')
        synth_query += f'{letters[idx]} {b} '
    
    print('\n--- RUNNING QUIZ AGENT SOLVER ---')
    res = solve_quiz_query(synth_query)
    print('Selected Answer:', res['selected_letter'], '->', res['selected_option'])
    print('Confidence:', res['confidence'])
    print('Probabilities:', res['probability_distribution'])
