# 1 downla

"""
Notes (from gdelt documentation)
The final dataset is a UTF8 JSON-NL file produced each minute (though at present typically those files will cluster in the minutes after every 15 minutes). Each line is a single article and contains the following fields:

    date. The timestamp the article was published. Roughly 30% of articles specify their exact publication timestamp, otherwise this is when we saw the article. Some articles may have older timestamps, which can mean we just saw them for the first time or older content was republished under a new URL.
    url. The URL of the article.
    domain. The full domain name of the URL.
    outletName. For the 71% of articles that include the full human-readable name of the news outlet (such as "The Wall Street Journal" instead of "wsj.com"), this field will contain that name, otherwise it will repeat the domain name.
    outletLogo. The thumbnail image to display as the logo of the news outlet that published the article. Around 87% of articles include this information and this image may not be the same as the standard "/favicon.ico" site image.
    outletTwitter. The Twitter handle of the news outlet that published the article. Around 49% of articles include this information, allowing the creation of rich display cards that connect users to the outlet's social media presence.
    title. The title of the article.
    image. The primary image of the article. Around 82% of articles specify an image to be displayed when sharing or linking to the article.
    desc. The contents of the "desc" metadata field. More than 91% of articles include this field. In some cases it may simply be the first sentence of the article, but in many cases it is a one-sentence summary that summarizes the general gist and focus of the article.
    lang. The language code returned by CLD2 for this article.
    author. Around 25% of articles specify the author(s) of the page. The format of this field differs widely across outlets and many contain multiple authors, outlet names, etc.

The dataset can be downloaded directly every minute as a JSON file with the following URL structure, with the date represented as "YYYYMMDDHHMMSS" in the UTC timezone. The first available file is seen below.
"""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from plumbum import FG, local


@dataclass(frozen=True, eq=True, order=True)
class DateTime:
    # minute-truncated datetime
    datetime: datetime

    @classmethod
    def from_ints(cls, year: int, month: int, day: int, hour: int, minute: int):
        return cls(datetime(year, month, day, hour, minute))

    def next(self):
        return DateTime(self.datetime + timedelta(minutes=1))

    def prev(self):
        return DateTime(self.datetime - timedelta(minutes=1))

    def minus(self, **kwargs):
        return DateTime(self.datetime - timedelta(**kwargs))

    @classmethod
    def now(cls):
        now = datetime.now()
        return DateTime(datetime(now.year, now.month, now.day, now.hour, now.minute))

    def as_str(self):
        return self.datetime.strftime("%Y%m%d%H%M00")


@dataclass
class NewsDownloader:
    base_path: Path

    def download_latest(self) -> tuple[DateTime, Path]:
        d = DateTime.now().minus(hours=2)
        while True:
            try:
                logging.info("Trying to download %s", d)
                fname = self.download(d)
                logging.info("Download succeeded")
                return d, fname
            except:
                logging.info("Download failed")
                time.sleep(1)
                d = d.prev()

    def download_latest_from(self, d: DateTime) -> tuple[DateTime, Path]:
        while True:
            try:
                logging.info("Trying to download %s", d)
                fname = self.download(d)
                logging.info("Download succeeded")
                return d, fname
            except:
                logging.info("Download failed")
                time.sleep(1)
                d = d.prev()

    def download(self, datetime: DateTime) -> Path:
        # download via plumbum / wget
        url = (
            f"http://data.gdeltproject.org/gdeltv3/gal/{datetime.as_str()}.gal.json.gz"
        )
        filename = self.base_path / f"{datetime.as_str()}.gal.json.gz"
        if filename.exists():
            return filename
        # TODO: run download
        try:
            wget = local["wget"]
            wget["-O", filename, url] & FG
            return filename
        except:
            if filename.exists():
                filename.unlink()
            raise

    def download_range(self, datetime_from: DateTime, datetime_to: DateTime):
        # download via plumbum / wget
        d = datetime_from
        while d <= datetime_to:
            self.download(d)
            d = d.next()

    def download_past_day(self):
        # download via plumbum / wget
        d, p = self.download_latest()
        datetime_to = DateTime.now().minus(hours=24)
        yield d, p
        d = d.prev()
        while d >= datetime_to:
            try:
                p = self.download(d)
                yield d, p
            except:
                pass
            d = d.prev()

    def download_past_week(self):
        """Download the past week's news files and yield their paths."""
        d, p = self.download_latest()
        datetime_to = DateTime.now().minus(days=7)
        yield d, p
        d = d.prev()
        while d >= datetime_to:
            p = self.download(d)
            yield d, p
            d = d.prev()
