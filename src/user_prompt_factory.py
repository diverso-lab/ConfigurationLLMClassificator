def get(row):
    # Construimos el texto de una instancia que le pasaremos al modelo.
    # Podemos meter aquí distinta información, ya sean textos o metadatos, 
    # en el formato que queramos (debemos indicarle los detalles al modelo
    # en el system prompt)
    return "Here you have the information about a bug report -> ID: " + row["Bug-ID"] +" \t Project: "+ row["Project"] +"\t Summary: "+ row["Summary"]+"\t Link: "+ row["Link"]+"\t Enviroment: "+ row["Enviroment"]