using CUDA

# cpu
f(x,y) = x^2 * y^2
input_cpu = [1e0,1e0]
derivative(temp -> f(temp[1],temp[2]),input_cpu,[1e0,0e0],2)

# gpu
f(x,y) = x^2 * y^2
input_gpu = CuArray([1e0,1e0])
direction_gpu = CuArray([1e0,0e0])
derivative(temp -> f(temp[1],temp[2]),input_gpu,direction_gpu,2)







