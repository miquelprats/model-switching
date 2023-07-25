from multiprocessing import Pool
import threading
import time
from flask import Flask, request
import torch
from torchvision import transforms
from PIL import Image
import lithops 
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, TimeoutError

MAX_ACTIVE_WORKERS = 5
num_active_workers = 0
resnet_models = ['resnet34', 'resnet50', 'resnet101']
num_workers = 2
request_queue = []

app = Flask(__name__)

fexec = lithops.FunctionExecutor()

def process_resnet_image(image_url, num_resnet):

    model = torch.hub.load('pytorch/vision:v0.10.0', resnet_models[num_resnet], pretrained=True)
    model.eval()
    response = requests.get(image_url)
    input_image = Image.open(BytesIO(response.content))
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    return num_resnet, output[0]

def execute_torch(args):
    image_url = args['value']
    args_sequence = [(image_url, i) for i in range(num_workers)]
    results = []
    time_limit = args['time_limit']
    start_time = args['start_time']
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures_torch = [executor.submit(process_resnet_image, image_url, num_resnet) for image_url, num_resnet in args_sequence]
        
        for future in futures_torch:
            try:
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_time = time_limit - elapsed_time
                
                #Checking request timeout
                if remaining_time <= 0:
                    for fut in futures_torch:
                        fut.cancel()
                    break
                
                #Automatic timeout
                result = future.result(timeout=10)
                results.append(result)
            except TimeoutError:
                results.append(None)

    results.sort(key=lambda x: x[0])

    result_of_highest_process = results[-1][1]
    return result_of_highest_process

def scheduler():
    global num_active_workers
    futures=[]
    while True:
        if request_queue and num_active_workers < MAX_ACTIVE_WORKERS:
            request = request_queue.pop(0)
            output_future = fexec.call_async(execute_torch, {'args': request})
            output_future.metadata = request  
            num_active_workers += 1
            futures.append(output_future)
        for fut in futures:
            if not fut.running:
                
                output = fexec.get_result(fut)
                
                probabilities = torch.nn.functional.softmax(output, dim=0)

                with open("imagenet_classes.txt", "r") as f:
                    categories = [s.strip() for s in f.readlines()]

                top5_prob, top5_catid = torch.topk(probabilities, 5)
                results=[]
                for i in range(top5_prob.size(0)):
                    results.append((categories[top5_catid[i]], top5_prob[i].item()))
                
                start_time = fut.metadata['start_time']
                total_time = time.time() - start_time
                print("Result:", results)
                print("Total time:", total_time, "segundos")
                futures.remove(fut)
                num_active_workers -= 1
        
        time.sleep(1)


@app.route('/afegir', methods=['POST'])
def add_request():
    image_request = request.json 
    image_request['start_time'] = time.time()  
    request_queue.append(image_request)
    return "Request added"

if __name__ == '__main__':
    scheduler_thread = threading.Thread(target=scheduler)
    scheduler_thread.start()
    app.run(debug=True)
