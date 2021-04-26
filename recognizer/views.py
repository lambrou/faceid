from django.shortcuts import render

def index(request):
    return render(request, 'recognizer/index.html', None)