import matlab.engine
eng = matlab.engine.start_matlab()
ret = eng.main(2,5)   #sending input to the function
print("Triarea : ")
print(ret)

eng.plotsin(nargout=0)
