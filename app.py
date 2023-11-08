import gradio as gr

iface = gr.Interface(
    fn=identify_species,
    inputs="directory",
    outputs="file",
    title="Wildlife Species Identification",
    description="Upload a directory containing wildlife images for species identification.",
    allow_flagging=False
)

iface.launch()
