

def get_file_name_desc(desc):
    name = ''
    for value in list(desc.values()):
        if isinstance(value,list):
            for element in value:
                name+=str(element)
        else : 
            name+=str(element)
    return name