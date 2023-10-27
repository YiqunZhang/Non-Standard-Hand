import gradio
import cv2
from component import bbox
from component import skeleton
from component import control
from component import ControlNetInpainter

image_example = cv2.cvtColor(cv2.imread("examples/a.png"), cv2.COLOR_BGR2RGB)

inpainter = ControlNetInpainter()
uni_height = 800

with gradio.Blocks() as interface:
    gradio.Markdown("# Step 1: Non-Standard Hand Detection")
    with gradio.Row():
        with gradio.Column(scale=2):
            input_1 = gradio.Image(type="numpy", label="Original Image", height=uni_height, value=image_example)
            input_2 = gradio.Checkbox(label="Bounding Box Mask Include Standard Hand")
            input_3 = gradio.Slider(minimum=0.5, maximum=2, step=0.01, value=1, label="Bounding Box Mask Expand Ratio")
            button_1 = gradio.Button("Submit")
        with gradio.Column(scale=2):
            output_1 = gradio.Textbox(label="Number of Hands & Classification with Confidence")
            output_2 = gradio.Image(type="numpy", label="Bounding Box", height=uni_height)
            output_3 = gradio.Image(type="numpy", label="Bounding Box Mask", height=uni_height)

        button_1.click(bbox, [input_1, input_2, input_3], [output_1, output_2, output_3])

    gradio.Markdown("# Step 2: Body Pose Estimation")
    with gradio.Row():
        with gradio.Column(scale=2):
            button_2 = gradio.Button("Submit")
        with gradio.Column(scale=2):
            output_4 = gradio.Textbox(label="Key Points String")
            output_5 = gradio.Image(type="numpy", label="Body Skeleton", height=uni_height)

        button_2.click(skeleton, [input_1], [output_4, output_5])

    gradio.Markdown("# Step 3: Control Image Generation")
    with gradio.Row():
        with gradio.Column(scale=2):
            input_4 = gradio.Radio(["opened-palm", "fist-back"], label="Hand Template", value="opened-palm")
            input_5 = gradio.Slider(minimum=0.5, maximum=2, step=0.01, value=1,
                                    label="Control Image and Mask Expand Ratio (From Wrist)")
            input_6 = gradio.Checkbox(label="Include Undetected Hand")
            button_3 = gradio.Button("Submit")

        with gradio.Column(scale=2):
            output_6 = gradio.Image(type="numpy", label="Visualization Image", height=uni_height)
            output_7 = gradio.Image(type="numpy", label="Combine Image", height=uni_height)
            output_8 = gradio.Image(type="numpy", label="Control Image", height=uni_height)
            output_9 = gradio.Image(type="numpy", label="Control Mask", height=uni_height)
            output_10 = gradio.Image(type="numpy", label="Union Mask", height=uni_height)

        button_3.click(control, [input_1, output_3, output_4, input_5, input_4, output_5, input_6],
                       [output_6, output_7, output_8, output_9, output_10])

    gradio.Markdown("# Step 4: ControlNet Inpainting")
    with gradio.Row():
        with gradio.Column(scale=2):
            button_4 = gradio.Button("Submit")
        with gradio.Column(scale=2):
            output_11 = gradio.Image(type="numpy", label="ControlNet Inpainting Image", height=uni_height)

        button_4.click(inpainter.inpainting, [input_1, output_10, output_8, input_4], [output_11])

    gradio.Markdown("# Step 5: IP2P Inpainting")
    with gradio.Row():
        with gradio.Column(scale=2):
            button_4 = gradio.Button("Submit")
        with gradio.Column(scale=2):
            output_12 = gradio.Image(type="numpy", label="IP2P Inpainting Image", height=uni_height)

        button_4.click(inpainter.ip2p, [output_11], [output_12])

interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
