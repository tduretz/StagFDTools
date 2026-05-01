to_tuple_if_vector(x) = x
to_tuple_if_vector(x::AbstractVector) = Tuple(x)

function struct2namedtuple(x::T) where T
    names = fieldnames(T)
    vals = ntuple(i -> getfield(x, i), Val(length(names)))
    return NamedTuple{names}(vals)
end

Base.@kwdef  struct Materials{T1, T2}
    plasticity          ::T1   = :none
    compressible        ::T2   = false
    g                   ::Vector{Float64} = [0.0, 0.0]
    ρ                   ::Vector{Float64} = Float64[]
    n                   ::Vector{Float64} = Float64[]
    η0                  ::Vector{Float64} = Float64[] 
    ξ0                  ::Vector{Float64} = Float64[]  
    G                   ::Vector{Float64} = Float64[] 
    C                   ::Vector{Float64} = Float64[] 
    ϕ                   ::Vector{Float64} = Float64[] 
    ηvp                 ::Vector{Float64} = Float64[] 
    β                   ::Vector{Float64} = Float64[] 
    ψ                   ::Vector{Float64} = Float64[] 
    B                   ::Vector{Float64} = Float64[] 
    cosϕ                ::Vector{Float64} = Float64[] 
    sinϕ                ::Vector{Float64} = Float64[] 
    sinψ                ::Vector{Float64} = Float64[]
    M                   ::Vector{Float64} = Float64[] # Golchin2021
    N                   ::Vector{Float64} = Float64[] # Golchin2021
    Pc                  ::Vector{Float64} = Float64[] # Golchin2021
    a                   ::Vector{Float64} = Float64[] # Golchin2021
    b                   ::Vector{Float64} = Float64[] # Golchin2021
    c                   ::Vector{Float64} = Float64[] # Golchin2021
    σT                  ::Vector{Float64} = Float64[] # Kiss2023 
    δσT                 ::Vector{Float64} = Float64[] # Kiss2023
    P1                  ::Vector{Float64} = Float64[] # Kiss2023
    τ1                  ::Vector{Float64} = Float64[] # Kiss2023
    P2                  ::Vector{Float64} = Float64[] # Kiss2023
    τ2                  ::Vector{Float64} = Float64[] # Kiss2023
end

function initialize_materials(n; compressible=false, plasticity=:none)
    materials = Materials(;
        compressible =  compressible,
        plasticity   =   plasticity,
        ρ            =       ones(n),
        n            =       ones(n), 
        η0           =       ones(n), 
        ξ0           =  1e50*ones(n),   
        G            =  1e50*ones(n), 
        β            = 1e-50*ones(n), 
        C            =  1e50*ones(n), 
        ϕ            =   0.0*ones(n), 
        ψ            =   0.0*ones(n), 
        ηvp          =   0.0*ones(n), 
        B            =       ones(n), 
        cosϕ         =       ones(n), 
        sinϕ         =   0.0*ones(n), 
        sinψ         =   0.0*ones(n),
        M            =   0.0*ones(n),
        N            =   0.0*ones(n),
        Pc           =   0.0*ones(n),
        a            =   0.0*ones(n),
        b            =   0.0*ones(n),
        c            =   0.0*ones(n),
        σT           =   0.0*ones(n), 
        δσT          =   0.0*ones(n),
        P1           =   0.0*ones(n),
        τ1           =   0.0*ones(n),
        P2           =   0.0*ones(n),
        τ2           =   0.0*ones(n),
    )
    return materials
end

function preprocess_materials(materials)
    @. materials.B    = (2 * materials.η0)^(-materials.n)
    @. materials.cosϕ = cosd.(materials.ϕ)
    @. materials.sinϕ = sind.(materials.ϕ)
    @. materials.sinψ = sind.(materials.ψ)
    return struct2namedtuple( materials )
end