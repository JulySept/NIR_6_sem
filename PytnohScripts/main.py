import subprocess

scripts = [
    #"prior.py",
    #"naive_bayes.py",
    #"linear_svm.py",
    #"ld_svm.py",
    "gradient_boosting.py",
    #"keras_model.py"
]

for script in scripts:
    print(f"\nЗапускается {script}...\n")
    subprocess.run(["python", script])
