
for data in dataloader:
	generated_text=model(input)
	generated_text=[0:5] #k probable output
	ground_sentence=sample["labels"] #with synonym, there can be many sentences 

	ed_final=0

	for pred_sentence in generated_text:
		for synonym_sentence in ground_sentence:
			ed=edit_distance(pred_sentence,synonym_sentence)
			if ed>ed_final:
				ed_final=ed
				best_sentence=pred_sentence

	save(best_sentence)



