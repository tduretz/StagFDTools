printxy(x) = display( rotr90(x[end:-1:1,end:-1:1]) )
av2D(x) = @. 0.25*(x[1:end-1,1:end-1] + x[2:end-0,1:end-1,] + x[1:end-1,2:end-0] + x[2:end-0,2:end-0])