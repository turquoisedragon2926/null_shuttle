include("config.jl")

using JLD2
using JUDI
using PyPlot
using Augmentor
using ImageFiltering
using ImageGather
using LinearAlgebra
using .Config

matplotlib.pyplot.switch_backend("Agg")

abstract type Objective end

function ContJitter(l::Number, num::Int)
    #l = length, num = number of samples
    interval_width = l / num
    interval_center = range(interval_width / 2, stop=l - interval_width / 2, length=num)
    randomshift = interval_width .* rand(Float32, num) .- interval_width / 2

    return interval_center .+ randomshift
end

struct FocusImageGatechObjective <: Objective
    params::Dict
    FocusImageGatechObjective() = new(Config.get_parameters())
end

function plot_cig(cig, n, d, offset_start, offset_end, n_offsets, filename)
    y = reshape(permutedims(cig, [2, 3, 1]), n[1], n[2], n_offsets, 1)

    PyPlot.rc("figure", titlesize=40)
    PyPlot.rc("font", family="serif")
    PyPlot.rc("xtick", labelsize=40)
    PyPlot.rc("ytick", labelsize=40)
    PyPlot.rc("axes", labelsize=40)     # Default fontsize for x and y labels
    PyPlot.rc("axes", titlesize=40)     # Default fontsize for titles

    ### X, Z position in km
    xpos = 3.6f3
    zpos = 2.7f3
    xgrid = Int(round(xpos / d[1]))
    zgrid = Int(round(zpos / d[2]))

    # Create a figure and a 2x2 grid of subplots
    fig, axs = subplots(2, 2, figsize=(20, 12), gridspec_kw=Dict("width_ratios" => [3, 1], "height_ratios" => [1, 3]))

    # Adjust the spacing between the plots
    subplots_adjust(hspace=0.0, wspace=0.0)

    vmin1, vmax1 = (-1, 1) .* quantile(abs.(vec(y[:, zgrid, :, 1])), 0.99)
    vmin2, vmax2 = (-1, 1) .* quantile(abs.(vec(y[:, :, div(n_offsets, 2)+1, 1])), 0.88)
    vmin3, vmax3 = (-1, 1) .* quantile(abs.(vec(y[xgrid, :, :, 1])), 0.999)
    sca(axs[1, 1])

    # Top left subplot
    axs[1, 1].imshow(y[:, zgrid, :, 1]', aspect="auto", cmap="gray", interpolation="none", vmin=vmin1, vmax=vmax1,
        extent=(0.0f0, (n[1] - 1) * d[1], offset_start, offset_end))
    axs[1, 1].set_ylabel("Offset [m]", fontsize=40)
    axs[1, 1].set_xticklabels([])
    axs[1, 1].set_xlabel("")
    hlines(y=0, colors=:b, xmin=0, xmax=(n[1] - 1) * d[1], linewidth=3)
    vlines(x=xpos, colors=:b, ymin=offset_start, ymax=offset_end, linewidth=3)

    # Bottom left subplot
    sca(axs[2, 1])
    axs[2, 1].imshow(y[:, :, div(n_offsets, 2)+1, 1]', aspect="auto", cmap="gray", interpolation="none", vmin=vmin2, vmax=vmax2,
        extent=(0.0f0, (n[1] - 1) * d[1], (n[2] - 1) * d[2], 0.0f0))
    axs[2, 1].set_xlabel("X [m]", fontsize=40)
    axs[2, 1].set_ylabel("Z [m]", fontsize=40)
    axs[2, 1].set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    axs[2, 1].set_xticklabels(["0", "1000", "2000", "3000", "4000", "5000"])
    axs[2, 1].set_yticks([1000, 2000, 3000])
    axs[2, 1].set_yticklabels(["1000", "2000", "3000"])

    axs[2, 2].get_shared_x_axes().join(axs[1, 1], axs[2, 1])
    vlines(x=xpos, colors=:b, ymin=0, ymax=(n[2] - 1) * d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=0, xmax=(n[1] - 1) * d[1], linewidth=3)

    # Top right subplot
    axs[1, 2].set_visible(false)

    # Bottom right subplot
    sca(axs[2, 2])
    axs[2, 2].imshow(y[xgrid, :, :, 1], aspect="auto", cmap="gray", interpolation="none", vmin=vmin3, vmax=vmax3,
        extent=(offset_start, offset_end, (n[2] - 1) * d[2], 0.0f0))
    axs[2, 2].set_xlabel("Offset [m]", fontsize=40)
    # Share y-axis with bottom left
    axs[2, 2].get_shared_y_axes().join(axs[2, 2], axs[2, 1])
    axs[2, 2].set_yticklabels([])
    axs[2, 2].set_ylabel("")
    vlines(x=0, colors=:b, ymin=0, ymax=(n[2] - 1) * d[2], linewidth=3)
    hlines(y=zpos, colors=:b, xmin=offset_end, xmax=offset_start, linewidth=3)

    # Remove the space between subplots and hide the spines
    for ax in reshape(axs, :)
        for spine in ["top", "right", "bottom", "left"]
            ax.spines[spine].set_visible(false)
        end
    end

    savefig(filename, bbox_inches="tight", dpi=300)
    close(fig)
end

function create_wavelet(timeD, dtD, f0)
    return ricker_wavelet(timeD, dtD, f0)
end

function generate_noise(d_obs, nsrc, snr, q)
    noise_ = deepcopy(d_obs)
    for l = 1:nsrc
        noise_.data[l] = randn(Float32, size(d_obs.data[l]))
        noise_.data[l] = real.(ifft(fft(noise_.data[l]) .* fft(q.data[1])))
    end
    noise_ = noise_ / norm(noise_) * norm(d_obs) * 10.0f0^(-snr / 20.0f0)
    return noise_
end

function create_geometry(n, d, nsrc, nxrec, dtD, timeD)
    xrec = range(0.0f0, stop=(n[1] - 1) * d[1], length=nxrec)
    yrec = 0.0f0
    zrec = range(d[1], stop=d[1], length=nxrec)
    recGeometry = Geometry(xrec, yrec, zrec; dt=dtD, t=timeD, nsrc=nsrc)

    ysrc = convertToCell(range(0.0f0, stop=0.0f0, length=nsrc))
    zsrc = convertToCell(range(d[1], stop=d[1], length=nsrc))
    xsrc = convertToCell(ContJitter((n[1] - 1) * d[1], nsrc))
    srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtD, t=timeD)

    return recGeometry, srcGeometry
end

function smooth_model(x; wb=10, down_rate=1)
    pl = ElasticDistortion(10 ÷ down_rate, 10 ÷ down_rate) ### feel free to define other distortion mechanisms
    xnew = augment(x, pl)
    xnew[:, 1:wb-1] .= minimum(x)
    xnew[:, wb:end] = 1.0f0 ./ Float32.(imfilter(1.0f0 ./ xnew[:, wb:end], Kernel.gaussian(10)))
    return xnew
end

function plot_velocity_model(x, n, d, filename)
    vmin = quantile(vec(x), 0.05)  # 5th percentile
    vmax = quantile(vec(x), 0.95)  # 95th percentile

    fig, ax = subplots(figsize=(20, 12))
    extentfull = (0.0f0, (n[1] - 1) * d[1], (n[end] - 1) * d[end], 0.0f0)
    cax = ax.imshow(x', vmin=vmin, vmax=vmax, extent=extentfull, aspect=0.45 * (extentfull[2] - extentfull[1]) / (extentfull[3] - extentfull[4]))
    ax.set_xlabel("X [m]", fontsize=40)
    ax.set_ylabel("Z [m]", fontsize=40)
    savefig(filename, bbox_inches="tight", dpi=300)
    close(fig)
end

function get_starting_model(φ::FocusImageGatechObjective)
    # Setup configurations
    n_offsets = φ.params["n_offsets"]
    offset_start = φ.params["offset_start"]
    offset_end = φ.params["offset_end"]
    f0 = φ.params["f0"]
    timeD = φ.params["timeD"]
    timeR = φ.params["timeR"]
    TD = φ.params["TD"]
    dtD = φ.params["dtD"]
    dtS = φ.params["dtS"]
    nbl = φ.params["nbl"]
    # n = φ.params["n"]
    d = φ.params["d"]
    o = φ.params["o"]
    nsrc = φ.params["nsrc"]
    # nxrec = φ.params["nxrec"]
    snr = φ.params["snr"]
    n_samples = φ.params["n_samples"]
    down_rate = φ.params["down_rate"]

    # data_path = "/slimdata/rafaeldata/fwiuq_eod/rtms_oed.jld2"
    # m_train = JLD2.jldopen(data_path, "r")["m_train"]

    # m0 = m_train[:, :, 1, 851]
    # n = (size(m_train, 1), size(m_train, 2))
    # nxrec = size(m_train, 1)

    data_path = "data/temp/m0.jld2"
    m0 = JLD2.jldopen(data_path, "r")["m0"]
    n = (size(m0, 1), size(m0, 2))
    nxrec = size(m0, 1)

    # Downsample everything
    f0 = f0 / down_rate
    m0 = m0[1:down_rate:end, 1:down_rate:end]
    d = d .* down_rate
    n = Int.(n ./ down_rate)
    nsrc = nsrc ÷ down_rate

    φ.params["model"] = Model(n, d, o, (1.0f0 ./ m0) .^ 2.0f0; nb=nbl / down_rate)

    title("Starting model")
    imshow(sqrt.(1.0f0 ./ m0)', cmap="GnBu", extent=(0, 10, 3, 0))
    xlabel("Lateral position [km]")
    ylabel("Depth [km]")
    savefig("starting_model_plot.png")

    # # Prepare model
    # m_mean = mean(m_train, dims=4)[1:down_rate:end, 1:down_rate:end, 1, 1]
    # wb = maximum(find_water_bottom(m_mean .- minimum(m_mean)))

    m_smooth = smooth_model(m0, wb=10, down_rate=down_rate)

    title("Smooth model")
    imshow(sqrt.(1.0f0 ./ m_smooth)', cmap="GnBu", extent=(0, 10, 3, 0))
    xlabel("Lateral position [km]")
    ylabel("Depth [km]")
    savefig("smooth_model_plot.png")

    ## Calculate d_obs
    wavelet = create_wavelet(timeD, dtD, f0)
    recGeometry, srcGeometry = create_geometry(n, d, nsrc, nxrec, dtD, timeD)

    model = Model(n, d, o, (1.0f0 ./ m_smooth) .^ 2.0f0; nb=nbl / down_rate)
    F = judiModeling(model, srcGeometry, recGeometry)
    q = judiVector(srcGeometry, wavelet)

    d_obs = F(1.0f0 ./ m_smooth .^ 2.0f0) * q
    noise_ = generate_noise(d_obs, nsrc, snr, q)
    d_obs = d_obs + noise_

    φ.params["F"] = F
    φ.params["q"] = q
    φ.params["d_obs"] = d_obs

    # Store downsamples parameters
    φ.params["dn"] = n
    φ.params["dd"] = d

    return m_smooth
end

function (φ::FocusImageGatechObjective)(x)

    n_offsets = φ.params["n_offsets"]
    offset_start = φ.params["offset_start"]
    offset_end = φ.params["offset_end"]
    d_obs = φ.params["d_obs"]
    F = φ.params["F"]
    q = φ.params["q"]

    offsetrange = range(offset_start, stop=offset_end, length=n_offsets)
    J = judiExtendedJacobian(F(1.0f0 ./ x .^ 2.0f0), q, offsetrange; options=Options(IC="isic"))
    cig = J' * d_obs
    
    n = φ.params["dn"]
    d = φ.params["dd"]

    plot_cig(cig, n, d, offset_start, offset_end, n_offsets, "test-cig.png")

    return norm(cig) # TODO: Update with focusing function
end

test = FocusImageGatechObjective()
m = get_starting_model(test)

test(m)
