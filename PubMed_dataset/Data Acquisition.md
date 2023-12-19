## Data Acquisition

We use ESearch to get all related PMID and store into the file named pubmed_ids_intelligence_2013_2023Id.txt

```sh
#!/bin/bash

QUERY="intelligence[Title/Abstract] AND (\"2013/01/01\"[Date - Publication] : \"2023/12/31\"[Date - Publication])"

esearch -db pubmed -query "$QUERY" |
efetch -format uid > pubmed_ids_intelligence_2013_2023Id.txt
```

Then we use efetch to download the xml format article according to the PMID.

```sh
command = f"efetch -db pubmed -id {pubmed_id} -format xml"
```

Finally, a library named lxml is adopt to parase the xml string and write the data we need into a new csv file.

The dataset can be downloaded: https://drive.google.com/file/d/1pkmay84d61XAeIGQAXR16bLTrsNYzmYF/view?usp=share_link