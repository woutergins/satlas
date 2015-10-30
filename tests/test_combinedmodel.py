import satlas.combinedmodel
import satlas.hfsmodel

def test_combinedmodel():
    I = 3.5
    J = [0, 1]
    ABC = [1000, 100, 10, 0, 0, 0]
    centroid = 0

    try:
        testmodels = [satlas.hfsmodel.HFSModel(I, J, ABC, centroid) for _ in range(3)]
        testobject = satlas.combinedmodel.CombinedModel(testmodels)
        if isinstance(testobject, satlas.combinedmodel.CombinedModel):
            testobject = testobject + testobject
            if isinstance(testobject, satlas.combinedmodel.CombinedModel):
                return True
            else:
                return False
        else:
            return False
    except:
        return False
