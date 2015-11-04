import satlas.basemodel

def test_basemodel():
    try:
        testobject = satlas.basemodel.BaseModel()
        if isinstance(testobject, satlas.basemodel.BaseModel):
            return True
        else:
            return False
    except:
        return False
