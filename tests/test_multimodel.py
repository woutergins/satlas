import satlas.multimodel
import satlas.hfsmodel

def test_combinedmodel():
    I = 3.5
    J = [0, 1]
    ABC = [1000, 100, 10, 0, 0, 0]
    centroid = 0

    try:
        testmodels = [satlas.hfsmodel.HFSModel(I, J, ABC, centroid) for _ in range(3)]
        testobject = sum(testmodels)
        if isinstance(testobject, satlas.multimodel.MultiModel):
            testobject = testmodels[0] + testmodels[1] + testmodels[2]
            if isinstance(testobject, satlas.multimodel.MultiModel):
                return True
            else:
                return False
        else:
            return False
    except:
        return False

print(test_combinedmodel())
