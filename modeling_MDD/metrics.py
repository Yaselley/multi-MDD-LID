from evaluate import load
import numpy as np
import difflib
from jiwer import compute_measures

class MDD_Evaluation:
    def __init__(self, refs, hyps, preds):
        self.refs = refs
        self.hyps = hyps
        self.preds = preds

    def sum(self, num, num2, list1, list2):
        sum = 0
        for i in range(len(list1)):
            if list1[i] == num and list2[i] == num2:
                sum += 1
        return sum

    def get_scores(self, ref, FA_seq):
        scores = []
        d = difflib.Differ()
        diff = d.compare(ref, FA_seq)  
        for i in diff:
            if i[0] == '-':
                scores.append(0)
            elif i[0] == ' ':
                scores.append(1)
        return scores

    def TA_unit(self, ref, hyp, pred):  
        result = self.get_scores(ref, pred)
        result_ = self.get_scores(ref, hyp)
        sum_ = self.sum(1, 1, result, result_)
        return sum_

    def FR_unit(self, ref, hyp, pred):
        result = self.get_scores(ref, pred)
        result_ = self.get_scores(ref, hyp)
        sum_ = self.sum(0, 1, result, result_)
        return sum_
    
    def FA_unit(self, ref, hyp, pred):
        result = self.get_scores(ref, pred)
        result_ = self.get_scores(ref, hyp)
        sum_ = self.sum(1, 0, result, result_)
        return sum_
    
    def TR_unit(self, ref, hyp, pred):
        result = self.get_scores(ref, pred)
        result_ = self.get_scores(ref, hyp)
        result_2 = self.get_scores(hyp, pred)

        sum_ = self.sum(0, 0, result, result_)
        sum_DE = self.sum(0, 0, result_2, result_2)
        sum_CD = self.sum(1, 1, result_2, result_2)
        return sum_, sum_DE, sum_CD

    def metrics_mdd(self):
        TA, FR, FA, TR, N, DE, CD = 0, 0, 0, 0, 0, 0, 0
        for i, elem in enumerate(self.refs):
            tmp_ref = elem
            tmp_hyp = self.hyps[i]

            N += len(tmp_hyp.split(' '))
            tmp_pred = self.preds[i]
            
            TA += self.TA_unit(tmp_ref, tmp_hyp, tmp_pred)
            FR += self.FR_unit(tmp_ref, tmp_hyp, tmp_pred)
            FA += self.FA_unit(tmp_ref, tmp_hyp, tmp_pred)
            TR += self.TR_unit(tmp_ref, tmp_hyp, tmp_pred)[0]
            DE += self.TR_unit(tmp_ref, tmp_hyp, tmp_pred)[1]
            CD += self.TR_unit(tmp_ref, tmp_hyp, tmp_pred)[2]
    
        
        cer_output = compute_measures(truth=[' '.join([*''.join(elem.split(' '))]) for elem in self.refs], hypothesis=[' '.join(elem.split(' ')) for elem in self.hyps])
        correct = N - cer_output['substitutions'] - cer_output['deletions']
        accuracy = N - cer_output['substitutions'] - cer_output['deletions'] - cer_output['insertions']
        
        correct = correct/N
        accuracy = accuracy/N
        
        if TR+FR !=0:
            precision = TR/(TR+FR)
        else :
            precision = 0
        
        if TR+FA !=0:
            recall = TR/(TR+FA)
        else:
            recall = 0
        
        if precision+recall !=0 :  
            F1 = 2*precision*recall/(precision+recall)
        else:
            F1 =0
            
        detection_accuracy = (TA+TR)/(TA+FR+FA+TR)
        diagnosis_accuracy = 1- (DE/(CD+DE))
        
        result = {'TA':TA, 'FR':FR, 'FA':FA, 'TR':TR, 'precision':precision, 'recall':recall, 'F1':F1, 'detection accuracy': detection_accuracy, 'diagnosis accuracy': diagnosis_accuracy}
        return result


#%%

import csv
class ComputeMetrics:
    def __init__(self, logits, targets, refs, tokenizer):
        self.tokenizer = tokenizer
        self.logits = logits
        self.targets = targets
        self.refs = refs

        mask =  self.targets >=0
        self.labels_pr_masked =  self.targets[mask]
        self.wer = load("wer")
        self.cer = load("cer")
                
        self.phone_preds = self.tokenizer.batch_decode(np.argmax(self.logits.transpose(1, 0, 2), axis=-1))
        print(self.phone_preds)
        print(self.logits.transpose(1, 0, 2).shape)
        self.phone_targets = self.tokenizer.batch_decode(self.targets)
        self.phone_refs = self.tokenizer.batch_decode(self.refs)
        self.phone_targets = [elem.replace('[UNK]','') for elem in self.phone_targets]
        self.phone_refs = [elem.replace('[UNK]','') for elem in self.phone_refs]

        self.mdd_eval = MDD_Evaluation(refs=self.phone_refs, hyps=self.phone_targets, preds=self.phone_preds)

        
    def results(self, save_file): 
        _wer = self.wer.compute(predictions= self.phone_preds, references= self.phone_targets)
        _cer = self.cer.compute(predictions= self.phone_preds, references= self.phone_targets)
        result = self.mdd_eval.metrics_mdd()

        with open(save_file, 'w', newline='') as csvfile:
            # create a writer object
            writer = csv.writer(csvfile)
            # write the header row
            writer.writerow(['targets', 'predicts', 'refs'])
            # iterate over the two lists and write each row
            for target, predict, ref in zip(self.phone_targets, self.phone_preds, self.phone_refs):
                writer.writerow([target, predict, ref])
        
        result['wer'] = _wer
        result['cer'] = _cer
        
        return result
