

def get_file_name_desc(desc):
    name = ''
    for value in list(desc.values()):
        if isinstance(value,list):
            for element in value:
                name+='_'+str(element)
        else : 
            name+='_'+str(value)
    return name