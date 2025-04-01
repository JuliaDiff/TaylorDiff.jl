using Symbolics, SymbolicUtils
using Graphs, GraphMakie, CairoMakie, LayeredLayouts
using Printf

function generate_dag(rexs...)
    exs = map(Symbolics.unwrap, rexs)
    dag = SimpleDiGraph()
    vertex_map = IdDict{Any, Int}()
    label_map = Dict{Int, String}()
    unicode_replace = Dict(
        "-" => "−",
        "*" => "×",
        "/" => "÷"
    )
    function dfs(node)
        if haskey(vertex_map, node)
            return vertex_map[node]
        end
        add_vertex!(dag)
        v = nv(dag)
        vertex_map[node] = v
        if iscall(node)
            op = operation(node)
            args = map(dfs, arguments(node))
            for arg in args
                add_edge!(dag, v, arg)
            end
            rop = repr(op)
            label_map[v] = get(unicode_replace, rop, rop)
        elseif node isa Number
            label_map[v] = @sprintf "%.1g" node
        else
            label_map[v] = repr(node)
        end
        return v
    end
    for ex in exs
        dfs(ex)
    end
    return dag, label_map
end

function plot_dag(dag, label_map)
    xs, ys, paths = solve_positions(Zarate(), dag)
    for (key, value) in paths
        paths[key] = (value[2], -value[1])
    end
    lay = Point.(zip(ys, -xs))
    wp = [Point2f.(zip(paths[e]...)) for e in edges(dag)]
    fig, ax, p = graphplot(
        dag; layout = lay, ilabels = [label_map[v] for v in 1:nv(dag)],
        waypoints = wp, node_color = :white)
    hidedecorations!(ax)
    hidespines!(ax)
    fig
end

@variables a b c d
x = (a + b) * (c + d)
y = (a - b) * (c - d)
z = (a + b) * (c - d)
w = (a - b) * (c + d)
p = x + y
q = z - w
rex = (x + y) / (z - w)

dag, label_map = generate_dag(p, q)
fig = plot_dag(dag, label_map)
save("expression_dag.png", fig)
