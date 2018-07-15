__author__ = 'Licheng'
from pprint import pprint
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider


class StimgidsEvaluator:
	def __init__(self, vist_sis, preds):
		"""
		:params vist_sis: vist's Story_in_Sequence instance
		:params preds   : [{'story_id', 'stimgids', 'pred_story_str'}]
		"""
		self.vist_sis = vist_sis
		self.eval_overall = {}   	# overall score
		self.eval_stimgids  = {}    # score on each story_imgids
		self.stimgids_to_eval = {}  # story_imgs -> eval
		self.preds = preds 			# [{story_id, stimgids, pred_story_str}]

	def evaluate(self, measure=None):
		"""
		measure is a subset of ['bleu', 'meteor', 'rouge', 'cider']
		if measure is None, we will apply all the above.
		"""

		# story_img_ids -> pred story str
		stimgids_to_Res = {item['stimgids']: [item['pred_story_str'].encode('ascii', 'ignore').decode('ascii')]
						for item in self.preds }

		# story_img_ids -> gt storie str(s)
		stimgids_to_stories = {}
		for story in self.vist_sis.stories:
			story_img_ids = '_'.join([str(img_id) for img_id in story['img_ids']])
			if story_img_ids in stimgids_to_stories:
				stimgids_to_stories[story_img_ids] += [story]
			else:
				stimgids_to_stories[story_img_ids] = [story]

		stimgids_to_Gts = {}
		for stimgids in stimgids_to_Res.keys():
			gd_story_strs = []
			related_stories = stimgids_to_stories[stimgids]
			for story in related_stories:
				gd_sent_ids = self.vist_sis.Stories[story['id']]['sent_ids']
				gd_story_str = ' '.join([self.vist_sis.Sents[sent_id]['text'] for sent_id in gd_sent_ids])
				gd_story_str = gd_story_str.encode('ascii', 'ignore').decode('ascii')  # ignore some weird token
				gd_story_strs += [gd_story_str]
			stimgids_to_Gts[stimgids] = gd_story_strs

		# tokenize
		# print 'tokenization ... '
		# tokenizer = PTBTokenizer()
		# self.stimgids_to_Res = tokenizer.tokenize(stimgids_to_Res)
		# self.stimgids_to_Gts = tokenizer.tokenize(stimgids_to_Gts)
		self.stimgids_to_Res = stimgids_to_Res
		self.stimgids_to_Gts = stimgids_to_Gts

		# =================================================
		# Set up scorers
		# =================================================
		print 'setting up scorers...'
		scorers = []
		if not measure:
			scorers = [
				(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
				(Meteor(),"METEOR"),
				(Rouge(), "ROUGE_L"),
				(Cider(), "CIDEr")
			]
		else:
			if 'bleu' in measure:
				scorers += [(Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"])]
			if 'meteor' in measure:
				scorers += [(Meteor(),"METEOR")]
			if 'rouge' in measure:
				scorers += [(Rouge(), "ROUGE_L")]
			if 'cider' in measure:
				scorers += [(Cider(), "CIDEr")]

		# =================================================
		# Compute scores
		# =================================================
		for scorer, method in scorers:
			print 'computing %s score ...' % (scorer.method())
			score, scores = scorer.compute_score(self.stimgids_to_Gts, self.stimgids_to_Res)
			if type(method) == list:
				for sc, scs, m in zip(score, scores, method):
					self.setEval(sc, m)
					self.setStimgidsToEval(scs, self.stimgids_to_Gts.keys(), m)
					print '%s: %.3f' % (m, sc)
			else:
				self.setEval(score, method)
				self.setStimgidsToEval(scores, self.stimgids_to_Gts.keys(), method)
				print '%s: %.3f' % (method, score)

		self.setEvalStimgids()

	def setEval(self, score, method):
		self.eval_overall[method] = score

	def setStimgidsToEval(self, scores, stimgids_list, method):
		for stimgids, score in zip(stimgids_list, scores):
			if not stimgids in self.stimgids_to_eval:
				self.stimgids_to_eval[stimgids] = {}
				self.stimgids_to_eval[stimgids]['stimgids'] = stimgids
			self.stimgids_to_eval[stimgids][method] = score

	def setEvalStimgids(self):
		self.eval_stimgids = [eval for stimgids, eval in self.stimgids_to_eval.items()]


if __name__ == '__main__':

	import os.path as osp
	from pprint import pprint
	import sys
	sys.path.insert(0, '../vist_api')
	from vist import Story_in_Sequence

	sis = Story_in_Sequence('/playpen/data/vist/images256', '/playpen/data/vist/annotations')

	# fake a predictions
	preds = []
	for story in sis.stories[:1]:
		img_ids = story['img_ids']
		stimgids = '_'.join(img_ids)  # knowing img_id is string beforehand
		pred_story_str = ' '.join([sis.Sents[sent_id]['text'] for sent_id in story['sent_ids']])
		preds += [{'stimgids': stimgids, 'pred_story_str': pred_story_str}]

	# load StimgidsEval
	seval = StimgidsEvaluator(sis, preds)
	seval.evaluate(['meteor'])










