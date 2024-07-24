# Poduct Hunt hotness binary prediction

Set up environment variables *AWS_ACCESS_KEY_ID* and *AWS_SECRET_ACCESS_KEY* for *Amazon S3* access.


## Scrapper

`PHScrapper.ipynb` notebook to scrap all products by single day.


## Service

Simple Flask API with `/predict [POST]` endpoint.


Body example:
```
{
  "url": "https://www.producthunt.com/posts/intelligent-canvas"
}
```

Response example:
```
{
    "prediction": "Hot"
}
```
```

## AirFlow


*DAG* for publishing/updating csv file on *Amazon S3 Bucket*.
Setup *S3 Connection* in AirFlow first

## Baseline


Fine-tune BERT on Product Hunt data. Fully-connected layer added to produce binary result.
