import argparse
import gradio as gr
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
from func import inpaint_predict, outpaint_predict, sketch_predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    return parser.parse_args()



def main():
    gr.close_all()
    args = parse_args()

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown('# Inpainting')
        with gr.Row():
            
            with gr.Column():
            
                with gr.Row():
                    original_image = gr.Image(source = 'upload', tool = 'sketch', type = 'pil'),
                with gr.Row():
                    text_prompt = gr.Textbox(label = 'prompt')
                with gr.Row():
                    seed = gr.Slider(0, 2022, step=1, value=2022, label="Seed")
                with gr.Row():
                    num_images = gr.Slider(1, 49, step=1, value=4, label="No. of Images")
                with gr.Row(variant='panel'):
                    run_button = gr.Button('Run')
            
            with gr.Column():
                with gr.Group():
                    with gr.Tabs():
                        with gr.TabItem("Output (Grid View)"):
                            result_grids = gr.Image(show_label=False)
                        with gr.TabItem("Output (Gallery)"):
                            result_gallery = gr.Gallery(show_label=False)
            
        run_button.click(fn=inpaint_predict,
                         inputs=[
                             original_image,
                             text_prompt,
                             seed,
                             num_images
                         ],
                         outputs=[
                             result_grids,
                             result_gallery
                         ])
        
        gr.Markdown('# Outpainting')
        with gr.Row():
            with gr.Column():
            
                with gr.Row():
                    original_image = gr.Image(source = 'upload', type = 'pil'),
                with gr.Row():
                    text_prompt = gr.Textbox(label = 'prompt')
                with gr.Row():
                    seed = gr.Slider(0, 2022, step=1, value=2022, label="Seed")
                with gr.Row():
                    num_images = gr.Slider(1, 49, step=1, value=4, label="No. of Images")
                with gr.Row(variant='panel'):
                    run_button = gr.Button('Run')
            
            with gr.Column():
                with gr.Group():
                    with gr.Tabs():
                        with gr.TabItem("Output (Grid View)"):
                            result_grids = gr.Image(show_label=False)
                        with gr.TabItem("Output (Gallery)"):
                            result_gallery = gr.Gallery(show_label=False)
            
        run_button.click(fn=outpaint_predict,
                         inputs=[
                             original_image,
                             text_prompt,
                             seed,
                             num_images
                         ],
                         outputs=[
                             result_grids,
                             result_gallery
                         ])
        
        
        gr.Markdown('# Sketch')
        with gr.Row():
            with gr.Column():
            
                with gr.Row():
                    original_image = gr.Image(source = 'canvas', type = 'pil'),
                with gr.Row():
                    text_prompt = gr.Textbox(label = 'prompt')
                with gr.Row():
                    seed = gr.Slider(0, 2022, step=1, value=2022, label="Seed")
                with gr.Row():
                    num_images = gr.Slider(1, 49, step=1, value=4, label="No. of Images")
                with gr.Row(variant='panel'):
                    run_button = gr.Button('Run')
            
            with gr.Column():
                with gr.Group():
                    with gr.Tabs():
                        with gr.TabItem("Output (Grid View)"):
                            result_grids = gr.Image(show_label=False)
                        with gr.TabItem("Output (Gallery)"):
                            result_gallery = gr.Gallery(show_label=False)
            
        run_button.click(fn=sketch_predict,
                         inputs=[
                             original_image,
                             text_prompt,
                             seed,
                             num_images
                         ],
                         outputs=[
                             result_grids,
                             result_gallery
                         ])
            
    demo.launch(
        server_name="0.0.0.0",
        server_port=7800,
        enable_queue=True,
        share=args.share,
    )


if __name__ == "__main__":
    main()
