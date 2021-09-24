

for i in range(int(input("how many queries"))):
    exec(open("ScrapeImages.py").read()) # need to store the queries and store in order to use TF data pipelines (labels and dirs)


exec(open("DataPipelineForTF.py").read()) # need to find way to pass data into scripts*
