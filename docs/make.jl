using NeuralLyapunov
using NeuralLyapunovProblemLibrary
using Documenter, DocumenterCitations

DocMeta.setdocmeta!(
    NeuralLyapunov, :DocTestSetup, :(using NeuralLyapunov); recursive = true)
DocMeta.setdocmeta!(
    NeuralLyapunovProblemLibrary,
    :DocTestSetup,
    :(
        using NeuralLyapunovProblemLibrary, ModelingToolkit, OrdinaryDiffEq, Random;
        Random.seed!(200)
    );
    recursive = true
)

MANUAL_PAGES = [
    "man.md",
    "man/pdesystem.md",
    "man/minimization.md",
    "man/decrease.md",
    "man/structure.md",
    "man/roa.md",
    "man/policy_search.md",
    "man/local_lyapunov.md"
]
DEMONSTRATION_PAGES = [
    "demos/damped_SHO.md",
    "demos/roa_estimation.md",
    "demos/policy_search.md",
    "demos/benchmarking.md"
]
NEURALLYAPUNOVPROBLEMLIBRARY_PAGES = [
    "lib.md",
    "lib/pendulum.md",
    "lib/double_pendulum.md",
    "lib/quadrotor.md"
]

bib = CitationBibliography(
    joinpath(@__DIR__, "ref.bib");
    style=:authoryear
)

makedocs(;
    modules = [NeuralLyapunov, NeuralLyapunovProblemLibrary],
    authors = "Nicholas Klugman <13633349+nicholaskl97@users.noreply.github.com> and contributors",
    sitename = "NeuralLyapunov.jl",
    format = Documenter.HTML(;
        canonical = "https://SciML.github.io/NeuralLyapunov.jl",
        edit_link = "master",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => vcat(MANUAL_PAGES, hide("man/internals.md")),
        "Demonstrations" => DEMONSTRATION_PAGES,
        "Test Problem Library" => NEURALLYAPUNOVPROBLEMLIBRARY_PAGES
    ],
    plugins=[bib]
)

deploydocs(;
    repo = "github.com/SciML/NeuralLyapunov.jl",
    devbranch = "master"
)
