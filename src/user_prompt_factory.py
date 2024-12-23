def get(row):
    # Build the text of an instance that we will pass to the model.
    # We can put different information here, whether texts or metadata,
    # in the format we want (we must tell the model the details on the system prompt).
    return "Here you have the information about a bug report -> ID: " + row["Bug-ID"] +" \t Project: "+ row["Project"] +"\t Summary: "+ row["Summary"]+"\t Link: "+ row["Link"]+"\t Enviroment: "+ row["Enviroment"]