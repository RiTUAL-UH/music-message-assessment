import os
import time
import json
import csv
from bs4 import BeautifulSoup
import requests
import pandas as pd
import pickle
import re


def clean_string(one_string):
    return re.sub(r"[^A-Za-z0-9]+", "", one_string.lower())


df = pd.read_csv("CSM_music_with_artist.csv", sep="\t")

page_list = []
for i in range(len(df)):
    music_name = df.iloc[i, 0]
    music_artist = df.iloc[i, 1]  # assume artist info is 2nd column
    artist_page = (
        "https://www.{your_lyrics_site}.com/"
        + music_artist[0].lower()
        + "/"
        + clean_string(music_artist)
        + ".html"
    )
    page_list.append(artist_page)


count = 0
for each in page_list:
    if requests.head(each).status_code != 404:
        count += 1


cookies = {
    "OptanonConsent": "xxx",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}

# get all lyrics of this artist from the lyrics page
response = requests.get(
    "https://www.{your_lyrics_site}.com/one_artist.html",
    headers=headers,
    cookies=cookies,
)

soup = BeautifulSoup(response.content)

soup.find_all(id="xxx")
