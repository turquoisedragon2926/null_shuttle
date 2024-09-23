using LinearAlgebra
using Zygote

# Abstract type with a type parameter for m_ref
abstract type Hypothesis end

struct DummyHypothesis <: Hypothesis end
struct ClosestToBackgroundHypothesis <: Hypothesis
    m_ref::AbstractArray
end

struct FarthestToBackgroundHypothesis <: Hypothesis
    m_ref::AbstractArray
end

struct ClosestToPointHypothesis <: Hypothesis
    m_ref::Number
end

struct FarthestToPointHypothesis <: Hypothesis
    m_ref::Number
end

function (ψ::ClosestToBackgroundHypothesis)(x::AbstractArray{T,N}) where {T, N}
    return norm(x - ψ.m_ref)
end

function (ψ::FarthestToBackgroundHypothesis)(x::AbstractArray{T,N}) where {T,N}
    return - norm(x - ψ.m_ref)

end

function (ψ::ClosestToPointHypothesis)(x::AbstractArray{T,N}) where {T,N}
    return norm(x - ψ.m_ref)
end

function (ψ::FarthestToPointHypothesis)(x::AbstractArray{T,N}) where {T,N}
    return - norm(x - ψ.m_ref)
end

function (ψ::DummyHypothesis)(x::AbstractArray{T,N}) where {T,N}
    return norm(x)
end

function init(ψ, φ, m)
    # Compute the gradient of φ with respect to m
    ∂φ∂m = Zygote.gradient(φ, m)[1]
    # Compute the gradient of ψ with respect to m
    ∂ψ∂m = Zygote.gradient(ψ, m)[1]

    # Define the orthogonal basis A
    n = length(m)
    A = spzeros(2n - 2, n)  # A sparse matrix of size (2n-2)x(n)

    # Populate the matrix A according to equation (17)
    j = 1
    for i in 1:n-1
        A[i, j] = ∂φ∂m[j+1]
        A[i, j+1] = -∂φ∂m[j]
        j += 1
    end

    # Compute x* = min_x ||Ax||_2 such that (Ax . ∂ψ∂m)
    x_star = A' * (A \ ∂ψ∂m)  # Pseudo-inverse approach

    # Calculate Δm_0
    Δm = A * x_star

    return Δm, ∂φ∂m
end
