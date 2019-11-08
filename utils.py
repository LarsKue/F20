def modify_data(modifier, *data):
    for d in data:
        yield type(d)(modifier(x) for x in d)