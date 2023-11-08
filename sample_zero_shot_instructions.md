For evaluating instruction models (i.e. chat-finetuned models) in zero-shot, prompt tuning can make a big difference for properly evaluating the capabilities of a model.

## Instructions from our experiments

For the experiments discussed in the Belebele paper, we evaluated multiple instruction models: Llama 2 chat (7B and 70B), GPT3.5-turbo, and BLOOMZ. In these evaluations, we instruct the model to provide the letter A, B, C, or D. We perform post-processing steps and accept answers predicted as e.g. (A) instead of A. We sometimes additionally remove the prefix `The correct answer is ` for predictions that do not start with one of the four accepted answers. The format of the instructions did change slightly across the models because some things worked better for some than others, but it was all just punctuation/minutia. 

The f-string generally looked like this:
```
f"{instruction}\n###\nPassage:\n{passage}\n###\nQuery:\n{query}\n###\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n###\nAnswer:\n"
```

Therefore, a full example would look like this:

```
Given the following passage, query, and answer choices, output the letter corresponding to the correct answer.
###
Passage:
Though many of the animals in the park are used to seeing humans, the wildlife is nonetheless wild and should not be fed or disturbed. According to park authorities, stay at least 100 yards/meters away from bears and wolves and 25 yards/meters from all other wild animals! No matter how docile they may look, bison, elk, moose, bears, and nearly all large animals can attack. Each year, dozens of visitors are injured because they didn't keep a proper distance. These animals are large, wild, and potentially dangerous, so give them their space. In addition, be aware that odors attract bears and other wildlife, so avoid carrying or cooking odorous foods and keep a clean camp.
###
Query:
Which of the following is not mentioned in the passage as a possible cause of wildlife attacks?
###
Choices:
(A) Strong smells
(B) Failure to maintain distance
(C) Feeding the wildlife
(D) Animals that are unfamiliar with humans
###
Answer:
```

## Proccessing the outputs:
Our response processing looked something like this, where we accepted 'A', '(A)', and some other closely related variants.

```python
correct = 0
for item in data[language]:
    qid = item['qid']
    
    answer = answers[language][qid].replace('(','').replace(')','')
    if answer not in ['A','B','C','D']:
        print("###############################")
        print("FAILED: ", answer)
        print("ACTUAL: ", item['answer'])
        answer = answer[0]  
    
    if item['answer'] == answer:
        correct += 1
        
print(correct/len(data[language]))
```
