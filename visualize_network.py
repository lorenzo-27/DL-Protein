from torchinfo import summary
# from torchview import draw_graph
# import graphviz

from rich.console import Console
console = Console()

def visualize(model, model_name, input_data):
    out = model(input_data)
    console.print(f'Computed output, shape = {out.shape=}')
    model_stats = summary(model,
                          input_data=input_data,
                          col_names=[
                              "input_size",
                              "output_size",
                              "num_params",
                              # "params_percent",
                              # "kernel_size",
                              # "mult_adds",
                          ],
                          row_settings=("var_names",),
                          col_width=18,
                          depth=8,
                          verbose=0,
                          )
    console.print(model_stats)
    # model_graph = draw_graph(model,
    #                          input_data=input_data,
    #                          expand_nested=True,
    #                          hide_module_functions=False,
    #                          depth=3)
    # graph = model_graph.visual_graph
    # svg = graphviz.Source(str(graph), format='svg')
    # svg.format = "svg"
    # svg.render(model_name)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('tboard')
    # writer.add_graph(transunet, X)
    # writer.close()
