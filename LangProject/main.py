import CustomLang

while True:
	text = input(' Dylang > ')
	result, error = CustomLang.run('<stdin>', text)

	if error: print(error.as_string())
	elif result: print(result)