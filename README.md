# model-switching
Simple program file to add requests to a flask app.
To use run the main.py
Add a request at the following url:
http://localhost:5000/afegir
The body will contain a value which is an url to an image to download and predict result using resnet
a number priority
and time limit which are seconds:
{
    value:"path/a/imatge",
    priority": 1,
    "time_limit": 15
}