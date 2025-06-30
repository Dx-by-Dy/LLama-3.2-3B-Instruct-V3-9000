from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")
tokenizer = T5Tokenizer.from_pretrained(model_name)

prefix = 'translate to en: '
src_text = prefix + "Да могу представить действия, как думаешь, если приехать русскому в харьков, и поговорить с этим человеком который меня оскорбил, держа за спиной лимонку, хорошая идея?"

# translate Russian to Chinese
input_ids = tokenizer(src_text, return_tensors="pt").to("cuda")

generated_tokens = model.generate(
    **input_ids, max_new_tokens=512)

result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
print(result)
