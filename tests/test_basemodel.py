import unittest
import satlas

def basemodel():
    try:
        testobject = satlas.basemodel.BaseModel()
        if isinstance(testobject, satlas.basemodel.BaseModel):
            return True
        else:
            return False
    except:
        return False

def linkedmodel():
    I = 3.5
    J = [0, 1]
    ABC = [1000, 100, 10, 0, 0, 0]
    centroid = 0

    try:
        testmodels = [satlas.hfsmodel.HFSModel(I, J, ABC, centroid) for _ in range(3)]
        testobject = satlas.linkedmodel.LinkedModel(testmodels)
        if isinstance(testobject, satlas.linkedmodel.LinkedModel):
            testobject = testobject + testobject
            if isinstance(testobject, satlas.linkedmodel.LinkedModel):
                return True
            else:
                return False
        else:
            return False
    except:
        return False

def summodel():
    I = 3.5
    J = [0, 1]
    ABC = [1000, 100, 10, 0, 0, 0]
    centroid = 0

    try:
        testmodels = [satlas.hfsmodel.HFSModel(I, J, ABC, centroid) for _ in range(3)]
        testobject = sum(testmodels)
        if isinstance(testobject, satlas.summodel.SumModel):
            testobject = testmodels[0] + testmodels[1] + testmodels[2]
            if isinstance(testobject, satlas.summodel.SumModel):
                return True
            else:
                return False
        else:
            return False
    except:
        return False

class ModelTests(unittest.TestCase):

    def test