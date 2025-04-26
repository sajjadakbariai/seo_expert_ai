#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
مدل هوش مصنوعی متخصص سئو (SEO Expert AI) - نسخه حرفه‌ای و تکمیل‌شده
توسعه‌دهنده: سجاد اکبری
ورژن: 5.1 (تکمیل و اصلاح شده)
تاریخ: 1403/02/18
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from transformers import TextDataset, DataCollatorForLanguageModeling
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse, urljoin
from urllib import robotparser
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import html
import time
from concurrent.futures import ThreadPoolExecutor # وارد شده اما در این نسخه استفاده نشده
import logging
from datetime import datetime
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union
import random
from colorama import init, Fore, Style
from dataclasses import dataclass, asdict
import hashlib
import pickle
from pathlib import Path

# Initialize colorama
init(autoreset=True)

# --- تنظیمات پایه لاگینگ ---
log_formatter = logging.Formatter(
    f'{Fore.GREEN}%(asctime)s{Fore.RESET} - {Fore.BLUE}%(name)s{Fore.RESET} - {Fore.YELLOW}%(levelname)s{Fore.RESET} - %(message)s'
)
log_file_handler = logging.FileHandler('seo_expert.log', encoding='utf-8')
log_file_handler.setFormatter(log_formatter)
log_console_handler = logging.StreamHandler()
log_console_handler.setFormatter(log_formatter)

logger = logging.getLogger("SEO_Expert_AI")
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
logger.addHandler(log_console_handler)
logger.propagate = False # جلوگیری از لاگ دوباره توسط root logger

# --- تنظیم seed برای تکرارپذیری ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# --- دانلود منابع مورد نیاز NLTK ---
def download_nltk_resources():
    try:
        logger.info("در حال بررسی و دانلود منابع NLTK (stopwords, punkt)...")
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        logger.warning("بسته stopwords یافت نشد. در حال دانلود...")
        nltk.download('stopwords')
    except LookupError:
         logger.warning("بسته stopwords یافت نشد (LookupError). در حال دانلود...")
         nltk.download('stopwords')

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        logger.warning("بسته punkt یافت نشد. در حال دانلود...")
        nltk.download('punkt')
    except LookupError:
        logger.warning("بسته punkt یافت نشد (LookupError). در حال دانلود...")
        nltk.download('punkt')
    logger.info("منابع NLTK آماده هستند.")

# --- کلاس تنظیمات پیشرفته ---
@dataclass
class SEOExpertConfig:
    # مدل
    MODEL_NAME: str = "HooshvareLab/gpt2-fa"
    MODEL_SAFE_NAME: str = MODEL_NAME.replace("/", "_")
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # آموزش
    BATCH_SIZE: int = 4 if DEVICE == "cuda" else 2 # کاهش یافته برای سازگاری بهتر با حافظه
    EPOCHS: int = 5 # کاهش یافته برای تست سریعتر، برای نتایج بهتر افزایش دهید
    LEARNING_RATE: float = 3e-5
    MAX_LENGTH: int = 512
    SAVE_STEPS: int = 1000
    LOGGING_STEPS: int = 100
    WARMUP_STEPS: int = 100 # کاهش یافته متناسب با EPOCHS کمتر
    WEIGHT_DECAY: float = 0.01

    # مسیرها
    BASE_DIR: str = "./seo_expert_ai"
    OUTPUT_DIR: str = f"{BASE_DIR}/models/{MODEL_SAFE_NAME}"
    REPORTS_DIR: str = f"{BASE_DIR}/reports"
    CACHE_DIR: str = f"{BASE_DIR}/cache"
    TRAIN_DATA_PATH: str = f"{BASE_DIR}/data/seo_train_data.json"
    TEST_DATA_PATH: str = f"{BASE_DIR}/data/seo_test_data.json"

    # پارامترهای تحلیل
    MIN_WORD_LENGTH: int = 3
    REQUEST_TIMEOUT: int = 20 # کمی افزایش یافت
    CACHE_EXPIRY: int = 86400 # 1 روز
    USER_AGENT: str = 'Mozilla/5.0 (compatible; SEOExpertBot/1.1; +ai.seokar.click)' # User agent سفارشی

    # دانش پایه سئو (در حال حاضر استفاده نشده، اما برای توسعه آینده مفید است)
    SEO_KNOWLEDGE_BASE: Dict = None

    def __post_init__(self):
        # ایجاد پوشه‌های مورد نیاز
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(f"{self.BASE_DIR}/data").mkdir(parents=True, exist_ok=True)

        # بارگذاری دانش پایه سئو
        self.SEO_KNOWLEDGE_BASE = {
            "technical_seo": ["سرعت سایت", "نمایه‌گذاری", "ساختار URL", "اسکیما مارکاپ", "بهینه‌سازی موبایل", "robots.txt", "sitemap.xml"],
            "on_page": ["محتوا", "کلمات کلیدی", "متا تگ‌ها", "ساختار محتوا", "لینک‌های داخلی", "بهینه سازی تصویر", "خوانایی"],
            "off_page": ["لینک‌سازی", "اعتبار دامنه", "شبکه‌های اجتماعی", "برندسازی", "سئو محلی"],
            "tools": ["Google Search Console", "Google Analytics", "Ahrefs", "SEMrush", "Moz", "Screaming Frog", "Lighthouse"],
            "algorithms": ["Panda", "Penguin", "Hummingbird", "BERT", "Core Web Vitals", "RankBrain", "E-A-T/E-E-A-T"]
        }
        logger.info(f"تنظیمات بارگذاری شد. دستگاه: {self.DEVICE}")

# --- سیستم کش ---
class SEOCache:
    # توجه: استفاده از pickle برای کش در محیط وب می‌تواند ریسک امنیتی داشته باشد.
    # برای یک ابزار CLI محلی، این ریسک کمتر است.
    def __init__(self, cache_dir: str, expiry: int = 86400):
        self.cache_dir = Path(cache_dir)
        self.expiry = expiry
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"سیستم کش در مسیر '{self.cache_dir}' با انقضای {self.expiry} ثانیه فعال شد.")

    def _get_cache_path(self, key: str) -> Path:
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self.cache_dir / f"{key_hash}.pkl"

    def get(self, key: str) -> Optional[object]:
        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            # logger.debug(f"Cache miss برای کلید: {key[:50]}...")
            return None

        try:
            # بررسی زمان انقضا
            if time.time() - cache_file.stat().st_mtime > self.expiry:
                logger.info(f"کش منقضی شده برای کلید: {key[:50]}... حذف شد.")
                cache_file.unlink()
                return None

            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            # logger.debug(f"Cache hit برای کلید: {key[:50]}...")
            return cached_data
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logger.error(f"خطا در خواندن یا باز کردن فایل کش '{cache_file}': {e}. فایل حذف می‌شود.")
            try:
                cache_file.unlink()
            except OSError:
                pass # اگر فایل وجود نداشت یا مشکل دیگری بود
            return None
        except Exception as e:
            logger.error(f"خطای ناشناخته در خواندن کش '{cache_file}': {e}")
            return None

    def set(self, key: str, value: object) -> None:
        cache_file = self._get_cache_path(key)
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            # logger.debug(f"کش ذخیره شد برای کلید: {key[:50]}...")
        except (pickle.PicklingError, OSError) as e:
            logger.error(f"خطا در ذخیره کش در فایل '{cache_file}': {e}")
        except Exception as e:
            logger.error(f"خطای ناشناخته در ذخیره کش '{cache_file}': {e}")

    def clear_cache(self) -> int:
        """حذف تمام فایل‌های کش."""
        count = 0
        try:
            for item in self.cache_dir.glob('*.pkl'):
                item.unlink()
                count += 1
            logger.info(f"{count} فایل کش با موفقیت حذف شد.")
            return count
        except Exception as e:
            logger.error(f"خطا در پاک کردن کش: {e}")
            return 0

# --- کلاس پردازش داده‌های سئو ---
class SEODatasetProcessor:
    def __init__(self, tokenizer, config: SEOExpertConfig, cache: SEOCache = None):
        self.tokenizer = tokenizer
        self.config = config
        self.cache = cache or SEOCache(config.CACHE_DIR, config.CACHE_EXPIRY)
        self._setup_persian_processing()
        logger.info("پردازشگر داده سئو مقداردهی اولیه شد.")

    def _setup_persian_processing(self):
        """آماده‌سازی منابع مورد نیاز برای پردازش فارسی"""
        try:
            self.persian_stopwords = set(stopwords.words('persian'))
            extra_stopwords = {
                'های', 'ترین', 'هایی', 'اند', 'بود', 'شد', 'شدن', 'خواهد', 'کنید',
                'کنیم', 'گیری', 'گذاری', 'بندی', 'کننده', 'شوند', 'کرده', 'یک', 'دو',
                'سه', 'با', 'به', 'از', 'در', 'بر', 'را', 'که', 'تا', 'آن', 'این', 'ها', 'می'
            }
            self.persian_stopwords.update(extra_stopwords)

            # الگوهای کامپایل شده برای کارایی بهتر
            self.persian_patterns = {
                'html_tags': re.compile(r'<.*?>'),
                'non_persian_alphanumeric': re.compile(r'[^\u0600-\u06FF\s\d]'), # اجازه به اعداد
                'extra_spaces': re.compile(r'\s+'),
                'punctuation': re.compile(r'[.,;:!؟،؛()\[\]{}«»<>]+'),
                'arabic_chars': re.compile(r'[ئيۀأإؤ]'), # برای نرمال سازی حروف عربی/فارسی
            }
            self.arabic_map = {'ي': 'ی', 'ئ': 'ی', 'ۀ': 'ه', 'أ': 'ا', 'إ': 'ا', 'ؤ': 'و'}

            logger.info("منابع پردازش فارسی آماده شد.")
        except Exception as e:
            logger.error(f"خطا در آماده‌سازی پردازش فارسی: {e}", exc_info=True)
            raise

    def _normalize_persian(self, text: str) -> str:
        """نرمال سازی حروف عربی/فارسی"""
        for k, v in self.arabic_map.items():
            text = text.replace(k, v)
        return text

    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """پاکسازی و نرمال‌سازی متن فارسی با قابلیت تنظیم سطح"""
        if not text:
            return ""
        cache_key = f"clean_text_{'agg_' if aggressive else ''}{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            # حذف تگ‌های HTML
            text = self.persian_patterns['html_tags'].sub(' ', text)
            # نرمال‌سازی حروف
            text = self._normalize_persian(text)
            # حذف نویسه‌های غیرفارسی (به جز اعداد و فاصله)
            text = self.persian_patterns['non_persian_alphanumeric'].sub('', text)
            # حذف علائم نگارشی
            text = self.persian_patterns['punctuation'].sub(' ', text)
            # حذف فاصله‌های اضافی
            text = self.persian_patterns['extra_spaces'].sub(' ', text).strip()

            # توکن‌سازی قبل از حذف استاپ‌وردها
            words = nltk.word_tokenize(text)

            # حذف استاپ‌وردها و کلمات کوتاه
            cleaned_words = [word for word in words
                             if word not in self.persian_stopwords
                             and len(word) >= self.config.MIN_WORD_LENGTH
                             and not word.isdigit()] # حذف اعداد خالص

            result = ' '.join(cleaned_words)

            if aggressive:
                 # در حالت aggressive می‌توان کارهای بیشتری کرد، مثلاً ریشه‌یابی (stemming)
                 # اما فعلاً تفاوت زیادی ایجاد نمی‌کنیم.
                 pass

            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"خطا در پاکسازی متن: {e}", exc_info=True)
            return text # بازگرداندن متن اصلی در صورت خطا

    def analyze_url(self, url: str) -> Optional[Dict]:
        """تحلیل پیشرفته URL با کش و مدیریت خطا"""
        if not url or not isinstance(url, str):
             return None
        cache_key = f"url_analysis_{hashlib.md5(url.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            parsed = urlparse(url)
            # بررسی اولیه برای اطمینان از حداقل اعتبار URL
            if not parsed.scheme or not parsed.netloc or parsed.scheme not in ['http', 'https']:
                logger.warning(f"URL نامعتبر یا بدون scheme/netloc: {url}")
                return None

            domain = parsed.netloc
            path = parsed.path if parsed.path else "/"
            query_params = parsed.query.split('&') if parsed.query else []
            param_count = len(query_params)

            # بررسی‌های سئو URL
            path_segments = [seg for seg in path.split('/') if seg]
            path_depth = len(path_segments)
            has_digits_in_path = any(c.isdigit() for c in path)
            has_underscore = '_' in path
            is_long_path = len(path) > 100
            has_utm = any(p.lower().startswith('utm_') for p in query_params)
            has_tracking = any(p.lower() in ['fbclid', 'gclid', 'msclkid'] for p in query_params)

            seo_friendly_score = 100
            issues = []
            if not (path == "/" or path.endswith('/') or '.' in path_segments[-1]):
                 seo_friendly_score -= 5
                 issues.append("URL فاقد اسلش انتهایی یا پسوند فایل است.")
            if path_depth > 4:
                 seo_friendly_score -= 10
                 issues.append(f"عمق مسیر URL زیاد است ({path_depth}).")
            if has_digits_in_path:
                 seo_friendly_score -= 5
                 issues.append("URL شامل اعداد در مسیر است.")
            if has_underscore:
                 seo_friendly_score -= 10
                 issues.append("URL شامل آندرلاین (_) است (خط تیره - ترجیح داده می‌شود).")
            if is_long_path:
                 seo_friendly_score -= 10
                 issues.append(f"طول مسیر URL زیاد است ({len(path)} کاراکتر).")
            if has_utm:
                 seo_friendly_score -= 5
                 issues.append("URL شامل پارامترهای UTM است.")
            if has_tracking:
                 seo_friendly_score -= 5
                 issues.append("URL شامل پارامترهای ردیابی (مانند fbclid, gclid) است.")
            if param_count > 2:
                 seo_friendly_score -= (param_count - 2) * 5
                 issues.append(f"تعداد پارامترهای کوئری زیاد است ({param_count}).")
            if parsed.scheme != 'https':
                seo_friendly_score -= 25
                issues.append("URL از HTTPS استفاده نمی‌کند.")

            result = {
                "url": url,
                "scheme": parsed.scheme,
                "domain": domain,
                "path": path,
                "query": parsed.query,
                "fragment": parsed.fragment,
                "is_https": parsed.scheme == 'https',
                "path_depth": path_depth,
                "param_count": param_count,
                "has_utm": has_utm,
                "has_tracking": has_tracking,
                "seo_friendly_score": max(0, seo_friendly_score),
                "seo_issues": issues,
                "is_root": path == "/",
            }

            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"خطا در تحلیل URL '{url}': {e}", exc_info=True)
            return None

    def extract_seo_elements(self, html_content: str, url: str = None) -> Dict:
        """استخراج عناصر سئو از HTML با تحلیل پیشرفته"""
        if not html_content:
            return {}
        # استفاده از هش محتوا برای کلید کش
        content_hash = hashlib.md5(html_content.encode('utf-8')).hexdigest()
        cache_key = f"seo_elements_{content_hash}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            base_url = urlparse(url).scheme + "://" + urlparse(url).netloc if url else None

            # --- عنوان ---
            title_tag = soup.find('title')
            title = title_tag.string.strip() if title_tag and title_tag.string else None
            title_length = len(title) if title else 0

            # --- توضیحات متا ---
            meta_description_tag = soup.find('meta', attrs={'name': re.compile(r'^description$', re.I)})
            meta_description = meta_description_tag.get('content', '').strip() if meta_description_tag else None
            meta_description_length = len(meta_description) if meta_description else 0

            # --- هدینگ‌ها ---
            headings = {}
            for level in range(1, 7):
                h_tags = soup.find_all(f'h{level}')
                headings[f'h{level}'] = {
                    'count': len(h_tags),
                    'texts': [tag.get_text(strip=True) for tag in h_tags]
                }

            # --- لینک‌ها ---
            internal_links_count = 0
            external_links_count = 0
            nofollow_links_count = 0
            all_links = soup.find_all('a', href=True)
            unique_hrefs = set()

            for link in all_links:
                href = link['href'].strip()
                if not href or href.startswith('#') or href.lower().startswith('javascript:') or href.lower().startswith('mailto:'):
                    continue

                # نرمال سازی URL های نسبی
                try:
                    absolute_href = urljoin(base_url, href) if base_url and not urlparse(href).scheme else href
                    parsed_href = urlparse(absolute_href)
                except ValueError:
                    continue # رد کردن URL های نامعتبر

                if absolute_href in unique_hrefs:
                     continue # رد کردن لینک های تکراری
                unique_hrefs.add(absolute_href)

                is_internal = False
                if base_url and parsed_href.netloc == urlparse(base_url).netloc:
                    is_internal = True
                    internal_links_count += 1
                elif not parsed_href.scheme and not parsed_href.netloc: # لینک های نسبی بدون base_url هم داخلی فرض می شوند
                    is_internal = True
                    internal_links_count += 1
                else:
                    external_links_count += 1

                if link.get('rel') and 'nofollow' in link.get('rel'):
                    nofollow_links_count += 1

            # --- تصاویر ---
            images = soup.find_all('img')
            images_with_alt = 0
            images_without_alt = 0
            missing_alt_srcs = []
            for img in images:
                alt_text = img.get('alt', '').strip()
                if alt_text:
                    images_with_alt += 1
                else:
                    images_without_alt += 1
                    missing_alt_srcs.append(img.get('src', 'Source not found'))

            # --- سایر عناصر ---
            canonical_tag = soup.find('link', attrs={'rel': 'canonical'})
            canonical_url = canonical_tag['href'].strip() if canonical_tag and canonical_tag.get('href') else None

            schema_tags = soup.find_all('script', type='application/ld+json')
            has_schema = len(schema_tags) > 0

            viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
            has_viewport = bool(viewport_tag and viewport_tag.get('content'))

            # زبان صفحه
            html_lang = soup.find('html').get('lang', 'نامشخص').lower() if soup.find('html') else 'نامشخص'

            result = {
                "title": title,
                "title_length": title_length,
                "meta_description": meta_description,
                "meta_description_length": meta_description_length,
                "headings": headings,
                "h1_count": headings.get('h1', {}).get('count', 0),
                "h1_texts": headings.get('h1', {}).get('texts', []),
                "images": {
                    "count": len(images),
                    "with_alt": images_with_alt,
                    "without_alt": images_without_alt,
                    "missing_alt_ratio": round(images_without_alt / len(images) * 100, 1) if images else 0,
                    #"missing_alt_srcs": missing_alt_srcs[:5] # فقط 5 تای اول برای جلوگیری از حجم زیاد
                },
                "links": {
                    "total_unique": len(unique_hrefs),
                    "internal": internal_links_count,
                    "external": external_links_count,
                    "nofollow": nofollow_links_count,
                },
                "canonical_url": canonical_url,
                "has_schema_markup": has_schema,
                "has_viewport_meta": has_viewport,
                "html_lang": html_lang
            }

            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"خطا در استخراج عناصر سئو: {e}", exc_info=True)
            return {} # بازگرداندن دیکشنری خالی در صورت خطا

    def calculate_readability_score(self, text: str) -> float:
        """محاسبه امتیاز خوانایی با فرمول پیشرفته فارسی"""
        if not text:
            return 0.0
        cache_key = f"readability_{hashlib.md5(text.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached is not None: # Check for None explicitly as 0.0 is a valid score
            return cached

        try:
            # استفاده از متن اصلی برای شمارش جملات بهتر است
            sentences = nltk.sent_tokenize(text)
            # استفاده از متن پاک شده برای شمارش کلمات معنی دار
            cleaned_text_for_words = self.clean_text(text, aggressive=False) # Aggressive زیاد لازم نیست
            words = nltk.word_tokenize(cleaned_text_for_words)

            num_sentences = len(sentences)
            num_words = len(words)

            if num_sentences == 0 or num_words == 0:
                return 0.0

            # محاسبه معیارهای لازم برای فرمول فلش-کینکیید (Flesch-Kincaid) - اقتباس شده
            # در فارسی معیار "سیلاب" به سادگی انگلیسی قابل محاسبه نیست. از طول کلمه به عنوان جایگزین استفاده می‌کنیم.
            avg_sentence_len = num_words / num_sentences
            # محاسبه میانگین طول کلمه (تعداد حروف)
            avg_word_len = sum(len(word) for word in words) / num_words if num_words > 0 else 0

            # فرمول فلش ریدینگ ایز فارسی (اقتباسی و تقریبی)
            # FRES = 206.835 - (1.015 * ASL) - (84.6 * AWL)
            # ASL = Average Sentence Length (میانگین طول جمله - تعداد کلمات)
            # AWL = Average Word Length (میانگین طول کلمه - تعداد حروف)
            # ضرایب ممکن است نیاز به تنظیم دقیق بر اساس داده‌های زبان فارسی داشته باشند.
            score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_word_len)

            # محدود کردن امتیاز بین 0 و 100
            score = max(0.0, min(100.0, score))

            self.cache.set(cache_key, score)
            return score
        except Exception as e:
            logger.error(f"خطا در محاسبه خوانایی: {e}", exc_info=True)
            return 0.0

    def prepare_dataset(self, data_path: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """آماده‌سازی داده‌های آموزشی با مدیریت خطا و کش"""
        data_path_obj = Path(data_path)
        if not data_path_obj.is_file():
            logger.error(f"فایل داده در مسیر '{data_path}' یافت نشد.")
            return None

        cache_key = f"dataset_{data_path_obj.stem}_{data_path_obj.stat().st_mtime}" # Include modification time in key
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"دیتاست از کش بارگذاری شد: {data_path}")
            return cached

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or not data:
                 logger.error(f"فرمت داده در '{data_path}' نامعتبر است یا فایل خالی است.")
                 return None

            inputs = []
            labels = [] # For language modeling, labels are usually same as inputs

            logger.info(f"شروع پردازش {len(data)} آیتم از فایل: {data_path}")
            for item in tqdm(data, desc=f"پردازش داده‌های {data_path_obj.name}"):
                try:
                    prompt = item.get("prompt")
                    response = item.get("response")

                    if not prompt or not isinstance(prompt, str) or not response or not isinstance(response, str):
                        logger.warning(f"آیتم نامعتبر یا ناقص در داده‌ها رد شد: {item}")
                        continue

                    # پاکسازی اولیه (نه خیلی شدید)
                    cleaned_prompt = self.clean_text(prompt, aggressive=False)
                    cleaned_response = self.clean_text(response, aggressive=False)

                    if not cleaned_prompt or not cleaned_response:
                        logger.warning(f"آیتم پس از پاکسازی خالی شد، رد شد: {item}")
                        continue

                    # فرمت ویژه برای آموزش مدل
                    formatted_text = (
                        f"[SEO_QUESTION] {cleaned_prompt}\n"
                        f"[SEO_ANSWER] {cleaned_response}\n"
                        f"[END]\n\n"
                    )

                    tokenized = self.tokenizer(
                        formatted_text,
                        max_length=self.config.MAX_LENGTH,
                        truncation=True,
                        padding='max_length', # پدینگ تا طول ماکسیمم
                        return_tensors="pt"
                    )

                    # اطمینان از اینکه توکن‌ها در اندازه مناسب هستند
                    if tokenized.input_ids.shape[-1] == self.config.MAX_LENGTH:
                        inputs.append(tokenized["input_ids"])
                        labels.append(tokenized["input_ids"].clone()) # Labels are same as inputs for LM
                    else:
                        logger.warning(f"آیتم پس از توکنیزه کردن طول نامناسب داشت، رد شد. Shape: {tokenized.input_ids.shape}")


                except Exception as e:
                    logger.warning(f"خطا در پردازش آیتم داده: {item}. خطا: {e}", exc_info=True)
                    continue

            if not inputs:
                logger.error("هیچ داده معتبری پس از پردازش یافت نشد.")
                return None

            # Concatenate tensors
            input_ids = torch.cat(inputs, dim=0)
            label_ids = torch.cat(labels, dim=0)

            logger.info(f"آماده‌سازی دیتاست کامل شد. تعداد نمونه‌های معتبر: {len(input_ids)}")
            result = (input_ids, label_ids)
            self.cache.set(cache_key, result)
            return result

        except json.JSONDecodeError as e:
            logger.error(f"خطا در خواندن یا تجزیه فایل JSON '{data_path}': {e}")
            return None
        except FileNotFoundError:
            logger.error(f"فایل داده در مسیر '{data_path}' یافت نشد.")
            return None
        except Exception as e:
            logger.error(f"خطای غیرمنتظره در آماده‌سازی دیتاست '{data_path}': {e}", exc_info=True)
            return None

# --- کلاس مدل هوش مصنوعی ---
class SEOExpertModel:
    def __init__(self, config: SEOExpertConfig):
        self.config = config
        # logger مجزا برای این کلاس (از logger اصلی استفاده می‌کند)
        self.logger = logging.getLogger(f"{logger.name}.SEOExpertModel")
        self.cache = SEOCache(config.CACHE_DIR, config.CACHE_EXPIRY)

        # بارگذاری توکنایزر و مدل
        self._load_tokenizer()
        self._load_model()

        # Initialize processor
        self.processor = SEODatasetProcessor(
            tokenizer=self.tokenizer,
            config=config,
            cache=self.cache
        )

        # Initialize tools
        self._setup_seo_tools()

        # Initialize robot parser
        self.robot_parser = robotparser.RobotFileParser()

        self.logger.info("مدل متخصص سئو با موفقیت مقداردهی اولیه شد.")


    def _load_tokenizer(self):
        """بارگذاری توکنایزر با مدیریت خطا"""
        try:
            self.logger.info(f"در حال بارگذاری توکنایزر از: {self.config.MODEL_NAME}")
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                self.config.MODEL_NAME,
                pad_token='[PAD]' # توکن پدینگ
            )

            # اضافه کردن توکن‌های ویژه برای سئو
            special_tokens = [
                '[SEO]', '[TECHNICAL]', '[ONPAGE]', '[OFFPAGE]',
                '[ANALYSIS]', '[RECOMMENDATION]', '[REPORT]',
                '[QUESTION]', '[ANSWER]', '[KEYWORD]', '[URL]', '[END]',
                '[SEO_QUESTION]', '[SEO_ANSWER]', '[SEO_EXPERT]' # توکن‌های فرمت‌بندی
            ]
            num_added_toks = self.tokenizer.add_special_tokens({
                'additional_special_tokens': special_tokens,
                'pad_token': '[PAD]' # اطمینان از تنظیم توکن پدینگ
            })
            self.logger.info(f"{num_added_toks} توکن ویژه به توکنایزر اضافه شد.")
            self.logger.info(f"اندازه واژگان توکنایزر: {len(self.tokenizer)}")

        except Exception as e:
            self.logger.critical(f"خطای مرگبار در بارگذاری توکنایزر: {e}", exc_info=True)
            raise RuntimeError(f"بارگذاری توکنایزر {self.config.MODEL_NAME} ناموفق بود.") from e

    def _load_model(self):
        """بارگذاری مدل با مدیریت خطا و انتقال به دستگاه مناسب"""
        try:
            model_path = Path(self.config.OUTPUT_DIR)
            model_to_load = None

            # 1. سعی در بارگذاری مدل fine-tuned شده از مسیر خروجی
            if (model_path / "config.json").exists() and (model_path / "pytorch_model.bin").exists():
                try:
                    self.logger.info(f"در حال بارگذاری مدل آموزش‌دیده از مسیر: {model_path}")
                    model_to_load = str(model_path)
                    self.model = GPT2LMHeadModel.from_pretrained(model_to_load)
                    self.logger.info("مدل آموزش‌دیده با موفقیت بارگذاری شد.")
                except Exception as e:
                    self.logger.warning(f"خطا در بارگذاری مدل آموزش‌دیده از '{model_path}': {e}. به مدل پایه بازمی‌گردیم.")
                    model_to_load = None # بازگشت به حالت پایه

            # 2. اگر مدل آموزش‌دیده نبود یا خطا داشت، مدل پایه را بارگذاری کن
            if model_to_load is None:
                self.logger.info(f"در حال بارگذاری مدل پایه از: {self.config.MODEL_NAME}")
                model_to_load = self.config.MODEL_NAME
                self.model = GPT2LMHeadModel.from_pretrained(model_to_load)
                # تغییر اندازه embedding ها متناسب با توکن‌های اضافه شده
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.logger.info("مدل پایه با موفقیت بارگذاری شد و اندازه embedding تنظیم شد.")

            # انتقال مدل به دستگاه (CPU/GPU)
            self.model.to(self.config.DEVICE)
            self.model.eval() # تنظیم مدل به حالت ارزیابی به طور پیش فرض
            self.logger.info(f"مدل به دستگاه '{self.config.DEVICE}' منتقل شد.")

        except Exception as e:
            self.logger.critical(f"خطای مرگبار در بارگذاری مدل: {e}", exc_info=True)
            raise RuntimeError(f"بارگذاری مدل {model_to_load or self.config.MODEL_NAME} ناموفق بود.") from e

    def _setup_seo_tools(self):
        """تنظیم ابزارهای تحلیل سئو"""
        self.seo_tools = {
            'content_analyzer': self.analyze_content,
            'technical_auditor': self.technical_seo_audit,
            'keyword_researcher': self.research_keywords, # Placeholder
            'competitor_analyzer': self.analyze_competitor, # Placeholder
            'trend_analyzer': self.analyze_trends # Placeholder
        }
        self.logger.info("ابزارهای سئو تنظیم شدند.")

    def train(self, train_data_path: str = None, test_data_path: str = None):
        """آموزش مدل با قابلیت‌های پیشرفته"""
        self.logger.info("شروع فرآیند آموزش مدل...")
        self.model.train() # تنظیم مدل به حالت آموزش

        try:
            train_data_path = train_data_path or self.config.TRAIN_DATA_PATH
            test_data_path = test_data_path or self.config.TEST_DATA_PATH

            self.logger.info("آماده‌سازی داده‌های آموزش...")
            train_data = self.processor.prepare_dataset(train_data_path)
            if train_data is None:
                self.logger.error("آماده سازی داده‌های آموزش ناموفق بود. آموزش لغو شد.")
                return None
            train_inputs, train_labels = train_data

            self.logger.info("آماده‌سازی داده‌های ارزیابی...")
            test_data = self.processor.prepare_dataset(test_data_path)
            if test_data is None:
                self.logger.warning("داده‌های ارزیابی یافت نشد یا نامعتبر بود. آموزش بدون ارزیابی ادامه می‌یابد.")
                eval_dataset = None
            else:
                test_inputs, test_labels = test_data
                eval_dataset = torch.utils.data.TensorDataset(test_inputs, test_labels)

            # ایجاد دیتاست PyTorch برای آموزش
            train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)

            # تنظیمات آموزش پیشرفته
            training_args = TrainingArguments(
                output_dir=self.config.OUTPUT_DIR,
                overwrite_output_dir=True,
                num_train_epochs=self.config.EPOCHS,
                per_device_train_batch_size=self.config.BATCH_SIZE,
                per_device_eval_batch_size=self.config.BATCH_SIZE,
                save_steps=self.config.SAVE_STEPS,
                save_total_limit=2, # ذخیره ۲ چک‌پوینت آخر
                logging_steps=self.config.LOGGING_STEPS,
                evaluation_strategy="steps" if eval_dataset else "no",
                eval_steps=self.config.SAVE_STEPS, # ارزیابی در همان مراحل ذخیره
                learning_rate=self.config.LEARNING_RATE,
                weight_decay=self.config.WEIGHT_DECAY,
                warmup_steps=self.config.WARMUP_STEPS,
                fp16=torch.cuda.is_available(), # استفاده از Mixed Precision در صورت وجود GPU
                gradient_accumulation_steps=2, # برای batch size موثر بزرگتر
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                greater_is_better=False,
                report_to="none", # غیرفعال کردن گزارش به سرویس‌های خارجی
                # logging_dir=f"{self.config.BASE_DIR}/logs", # فعال کردن در صورت نیاز به لاگ‌های تنسوربورد
            )

            # Data Collator برای مدل‌های زبانی
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False # ما در حال آموزش مدل LM هستیم نه MLM
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                # compute_metrics=compute_metrics, # می توان تابعی برای محاسبه متریک‌های بیشتر اضافه کرد
            )

            # شروع آموزش
            self.logger.info(f"شروع آموزش برای {self.config.EPOCHS} دوره...")
            train_result = trainer.train()

            # ذخیره مدل و نتایج نهایی
            self.logger.info("آموزش کامل شد. در حال ذخیره مدل نهایی و توکنایزر...")
            final_output_dir = Path(self.config.OUTPUT_DIR) / "final_model"
            trainer.save_model(str(final_output_dir))
            self.tokenizer.save_pretrained(str(final_output_dir))
            self.logger.info(f"مدل و توکنایزر نهایی در {final_output_dir} ذخیره شد.")

            # ذخیره متریک‌ها
            metrics = train_result.metrics
            metrics_file = Path(self.config.OUTPUT_DIR) / "training_metrics.json"
            try:
                with open(metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=4)
                self.logger.info(f"نتایج آموزش در {metrics_file} ذخیره شد.")
            except Exception as e:
                self.logger.error(f"خطا در ذخیره نتایج آموزش: {e}")

            self.model.eval() # بازگرداندن مدل به حالت ارزیابی
            self.logger.info("فرآیند آموزش با موفقیت به پایان رسید.")
            return metrics

        except Exception as e:
            self.logger.error(f"خطا در فرآیند آموزش: {e}", exc_info=True)
            self.model.eval() # اطمینان از بازگشت به حالت eval در صورت خطا
            return None # یا raise مجدد خطا بسته به نیاز

    def generate_seo_advice(self, prompt: str, max_length: int = 300, **kwargs) -> str:
        """تولید پاسخ تخصصی سئو با پارامترهای پیشرفته"""
        if not prompt:
            return "لطفاً یک سوال یا دستور معتبر وارد کنید."

        self.model.eval() # اطمینان از حالت ارزیابی
        # کش برای تولید پاسخ‌ها ممکن است خیلی مفید نباشد چون پرامپت‌ها معمولا متفاوتند
        # اما برای پرامپت‌های تکراری می‌تواند مفید باشد
        cache_key = f"seo_advice_prompt_{hashlib.md5(prompt.encode('utf-8')).hexdigest()}_maxlen_{max_length}"
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.info("پاسخ سئو از کش بازگردانده شد.")
            return cached

        try:
            # فرمت‌بندی پیشرفته پرسش با توکن‌های ویژه
            formatted_prompt = (
                f"[SEO_QUESTION] {prompt}\n"
                f"[SEO_EXPERT]" # مدل باید از اینجا شروع به تولید کند
            )

            # توکنیزه کردن ورودی
            input_ids = self.tokenizer.encode(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.MAX_LENGTH // 2 # محدود کردن طول پرامپت برای جا دادن پاسخ
            ).to(self.config.DEVICE)

            # پارامترهای پیشرفته تولید متن
            default_generation_config = {
                "max_length": input_ids.shape[1] + max_length, # طول کل = طول ورودی + حداکثر طول جدید
                "num_return_sequences": 1,
                "no_repeat_ngram_size": 3, # جلوگیری از تکرار ۳-گرام‌ها
                "early_stopping": True, # توقف زود هنگام وقتی توکن پایان تولید شود
                "temperature": 0.75, # کمی خلاقیت بیشتر
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True, # فعال کردن نمونه‌برداری
                "repetition_penalty": 1.2, # جریمه تکرار کلمات
                "length_penalty": 1.0, # جریمه طول (۱ یعنی بدون جریمه)
                "pad_token_id": self.tokenizer.eos_token_id, # استفاده از EOS به جای PAD برای توقف
                "eos_token_id": self.tokenizer.eos_token_id, # مشخص کردن توکن پایان
            }
            # ادغام با پارامترهای ورودی کاربر
            generation_config = {**default_generation_config, **kwargs}

            self.logger.info(f"در حال تولید پاسخ برای پرامپت: '{prompt[:100]}...'")
            with torch.no_grad(): # غیرفعال کردن محاسبه گرادیان برای سرعت بیشتر
                output_sequences = self.model.generate(
                    input_ids=input_ids,
                    **generation_config
                )

            # دیکد کردن توکن‌های خروجی
            decoded_output = self.tokenizer.decode(
                output_sequences[0], # فقط اولین دنباله خروجی
                skip_special_tokens=False, # توکن‌های ویژه را نگه دار برای پردازش
                clean_up_tokenization_spaces=True
            )

            # پردازش پاسخ برای نمایش بهتر
            response = self._process_response(decoded_output, formatted_prompt)
            self.cache.set(cache_key, response) # ذخیره پاسخ نهایی در کش
            self.logger.info(f"پاسخ تولید شد: '{response[:100]}...'")
            return response

        except Exception as e:
            self.logger.error(f"خطا در تولید پاسخ سئو برای پرامپت '{prompt[:50]}...': {e}", exc_info=True)
            return "متاسفانه در تولید پاسخ خطایی رخ داد. لطفاً دوباره تلاش کنید یا پرامپت را تغییر دهید."

    def _process_response(self, generated_text: str, original_prompt_formatted: str) -> str:
        """پردازش پاسخ تولید شده برای حذف پرامپت و توکن‌های اضافی"""
        try:
            # 1. حذف بخش پرامپت اصلی از متن تولید شده
            # گاهی مدل خود پرامپت را تکرار می‌کند
            response_part = generated_text
            if generated_text.startswith(original_prompt_formatted):
                 response_part = generated_text[len(original_prompt_formatted):]
            # راه دیگر: پیدا کردن اولین توکن پاسخ مورد انتظار
            elif "[SEO_EXPERT]" in generated_text:
                response_part = generated_text.split("[SEO_EXPERT]", 1)[-1]
            elif "[SEO_ANSWER]" in generated_text:
                 response_part = generated_text.split("[SEO_ANSWER]", 1)[-1]


            # 2. حذف توکن‌های ویژه اضافی و توکن پایان
            special_tokens_to_remove = [
                '[SEO]', '[TECHNICAL]', '[ONPAGE]', '[OFFPAGE]',
                '[ANALYSIS]', '[RECOMMENDATION]', '[REPORT]',
                '[QUESTION]', '[ANSWER]', '[KEYWORD]', '[URL]',
                '[SEO_QUESTION]', '[SEO_ANSWER]', '[SEO_EXPERT]',
                '[PAD]',
                str(self.tokenizer.eos_token), # توکن پایان
                '[END]' # توکن پایان سفارشی ما
            ]

            processed_response = response_part
            for token in special_tokens_to_remove:
                processed_response = processed_response.replace(token, '')

            # 3. نرمال‌سازی فاصله‌ها و خطوط جدید
            processed_response = re.sub(r'\s+', ' ', processed_response).strip()
            # تبدیل چند خط جدید پشت سر هم به دو خط جدید (برای پاراگراف‌بندی)
            processed_response = re.sub(r'\n\s*\n', '\n\n', processed_response)
            # حذف خطوط جدید اضافی در ابتدا و انتها
            processed_response = processed_response.strip('\n ')

            # اگر پاسخ خالی شد، یک پیام پیش‌فرض برگردان
            if not processed_response:
                return "پاسخ معتبری تولید نشد."

            return processed_response

        except Exception as e:
            self.logger.warning(f"خطا در پردازش پاسخ مدل: {e}. بازگرداندن متن خام.")
            return generated_text # بازگرداندن متن اصلی در صورت خطا

    def analyze_content(self, text: Optional[str] = None, url: Optional[str] = None) -> Dict:
        """تحلیل پیشرفته محتوا (از متن یا URL) با قابلیت‌های جامع"""
        if not text and not url:
            return {"error": "برای تحلیل محتوا، باید متن یا URL ارائه شود."}

        analysis_key_part = text if text else url
        cache_key = f"content_analysis_{hashlib.md5(analysis_key_part.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.info(f"نتایج تحلیل محتوا برای {'URL: '+url if url else 'متن'} از کش بازگردانده شد.")
            return cached

        try:
            content_to_analyze = text
            fetched_from_url = False
            if url:
                self.logger.info(f"در حال دریافت محتوا از URL برای تحلیل: {url}")
                # اینجا از تابع technical_seo_audit استفاده نمی‌کنیم تا فقط متن را بگیریم
                headers = {'User-Agent': self.config.USER_AGENT}
                try:
                    response = requests.get(url, headers=headers, timeout=self.config.REQUEST_TIMEOUT, allow_redirects=True)
                    response.raise_for_status()
                    # حدس زدن انکودینگ مناسب
                    response.encoding = response.apparent_encoding
                    html_content = response.text
                    soup = BeautifulSoup(html_content, 'html.parser')
                    # استخراج متن اصلی (ممکن است نیاز به بهبود داشته باشد)
                    body_tag = soup.find('body')
                    content_to_analyze = body_tag.get_text(separator='\n', strip=True) if body_tag else html_content
                    fetched_from_url = True
                    self.logger.info(f"محتوا از {url} با موفقیت دریافت شد.")
                except RequestException as e:
                    self.logger.error(f"خطا در دریافت محتوا از URL '{url}': {e}")
                    return {"error": f"خطا در دریافت محتوا از URL: {e}"}
                except Exception as e:
                     self.logger.error(f"خطای ناشناخته در پردازش محتوای URL '{url}': {e}")
                     return {"error": f"خطای ناشناخته در پردازش محتوای URL: {e}"}

            if not content_to_analyze:
                return {"error": "محتوایی برای تحلیل یافت نشد."}

            # تحلیل محتوای متنی
            # متن تمیز شده برای شمارش کلمات و کلمات کلیدی
            cleaned_text = self.processor.clean_text(content_to_analyze)
            words_list = cleaned_text.split()
            word_count = len(words_list)

            # متن اصلی برای خوانایی و شمارش جملات
            readability_score = self.processor.calculate_readability_score(content_to_analyze)
            readability_level = self._get_readability_level(readability_score)
            sentences = nltk.sent_tokenize(content_to_analyze)
            sentence_count = len(sentences)
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

            # تحلیل ساختاری (پاراگراف‌ها بر اساس خطوط جدید)
            paragraphs = [p for p in content_to_analyze.split('\n') if p.strip()]
            paragraph_count = len(paragraphs)

            # محاسبه تراکم کلمات کلیدی
            keyword_density = self._calculate_advanced_keyword_density(cleaned_text)

            # تحلیل احساس (بسیار ساده - نیاز به بهبود دارد)
            sentiment = self._analyze_sentiment(cleaned_text)

            # ساخت خلاصه تحلیل برای تولید توصیه
            analysis_summary = (
                f"تحلیل سئو محتوا {'برای URL: '+url if url else ''}:\n"
                f"- طول محتوا: {len(content_to_analyze)} کاراکتر، {word_count} کلمه\n"
                f"- تعداد جملات: {sentence_count}, میانگین طول جمله: {avg_sentence_length:.1f} کلمه\n"
                f"- تعداد پاراگراف‌ها: {paragraph_count}\n"
                f"- امتیاز خوانایی: {readability_score:.1f}/100 ({readability_level})\n"
                f"- احساس کلی: {sentiment}\n"
                f"- 5 کلمه کلیدی برتر: {', '.join([f'{k} ({v:.1f}%)' for k, v in list(keyword_density.items())[:5]])}\n"
                f"لطفاً توصیه‌هایی برای بهبود سئوی این محتوا ارائه دهید."
            )

            self.logger.info("در حال تولید توصیه‌های سئو برای محتوا...")
            recommendations = self.generate_seo_advice(
                analysis_summary,
                max_length=350, # طول بیشتر برای توصیه‌ها
                temperature=0.6 # کمی خلاقیت کمتر برای توصیه‌ها
            )

            result = {
                "source": f"URL: {url}" if url else "Input Text",
                "metrics": {
                    "char_count": len(content_to_analyze),
                    "word_count": word_count,
                    "sentence_count": sentence_count,
                    "paragraph_count": paragraph_count,
                    "avg_sentence_length": round(avg_sentence_length, 1),
                    "readability_score": round(readability_score, 1),
                    "readability_level": readability_level,
                    "sentiment": sentiment,
                    # "avg_word_length": round(sum(len(word) for word in words_list) / word_count if word_count else 0, 1),
                },
                "keyword_analysis": {k: round(v, 2) for k, v in keyword_density.items()}, # رند کردن درصدها
                "top_keywords": list(keyword_density.keys())[:10], # ۱۰ کلمه کلیدی برتر
                "recommendations": recommendations,
                "timestamp": datetime.now().isoformat()
            }

            # اگر از URL تحلیل شده، اطلاعات URL را هم اضافه کن
            if url and fetched_from_url:
                url_analysis_data = self.processor.analyze_url(url)
                if url_analysis_data:
                    result['url_analysis'] = url_analysis_data

            self.cache.set(cache_key, result)
            self.logger.info(f"تحلیل محتوا برای {'URL: '+url if url else 'متن'} کامل شد.")
            return result

        except Exception as e:
            self.logger.error(f"خطای ناشناخته در تحلیل محتوا: {e}", exc_info=True)
            return {"error": f"خطای ناشناخته در تحلیل محتوا: {str(e)}"}

    def _calculate_advanced_keyword_density(self, cleaned_text: str) -> Dict[str, float]:
        """محاسبه تراکم کلمات کلیدی با تحلیل پیشرفته‌تر (فرکانس ساده)"""
        # توجه: این روش هنوز ساده است و TF-IDF یا embedding را در نظر نمی‌گیرد.
        # صرفاً فراوانی کلمات غیر استاپ‌ورد را محاسبه می‌کند.
        try:
            words = [w for w in nltk.word_tokenize(cleaned_text)
                     if len(w) >= self.config.MIN_WORD_LENGTH and not w.isdigit()]

            if not words:
                return {}

            total_words = len(words)
            freq_dist = Counter(words)

            # محاسبه درصد تراکم ساده
            density = {word: (count / total_words) * 100 for word, count in freq_dist.items()}

            # مرتب‌سازی بر اساس تراکم و انتخاب ۲۰ کلمه برتر
            # مرتب سازی اولیه بر اساس فراوانی (count)، سپس تراکم برای نمایش
            # top_keywords = sorted(freq_dist.items(), key=lambda item: item[1], reverse=True)[:20]
            # top_keywords_density = {word: density[word] for word, count in top_keywords}

            # مرتب‌سازی نهایی بر اساس درصد تراکم
            sorted_density = dict(sorted(density.items(), key=lambda item: item[1], reverse=True)[:20])


            return sorted_density

        except Exception as e:
            self.logger.error(f"خطا در محاسبه تراکم کلمات کلیدی: {e}", exc_info=True)
            return {}

    def _analyze_sentiment(self, cleaned_text: str) -> str:
        """تحلیل احساس و تن محتوا (بسیار ساده - نیاز به بهبود جدی دارد)"""
        # هشدار: این پیاده‌سازی بسیار ساده و غیرقابل اتکا است.
        # برای نتایج بهتر باید از مدل‌های تحلیل احساسات آموزش دیده استفاده کرد.
        try:
            positive_words = {'عالی', 'خوب', 'بهترین', 'فوق العاده', 'مفید', 'کاربردی', 'مناسب', 'سریع', 'آسان', 'قدرتمند', 'حرفه ای', 'افزایش', 'بهبود'}
            negative_words = {'بد', 'ضعیف', 'نامناسب', 'کند', 'سخت', 'پیچیده', 'مشکل', 'کاهش', 'محدود', 'ایراد', 'خطا'}

            words = set(cleaned_text.split()) # استفاده از set برای سرعت بیشتر

            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))

            # تعیین احساس بر اساس تفاوت شمارش
            if pos_count > neg_count + 1: # نیاز به تفاوت بیشتر برای مثبت بودن
                return "احتمالاً مثبت"
            elif neg_count > pos_count:
                return "احتمالاً منفی"
            else:
                return "خنثی یا نامشخص"
        except Exception as e:
            self.logger.warning(f"خطا در تحلیل ساده احساسات: {e}")
            return "نامشخص"

    def _get_readability_level(self, score: float) -> str:
        """تعیین سطح خوانایی بر اساس امتیاز (مقیاس فلش-کینکیید)"""
        if score >= 90: return "خیلی آسان (مناسب مقطع ابتدایی)"
        if score >= 80: return "آسان (مناسب مقطع راهنمایی اول)"
        if score >= 70: return "نسبتاً آسان (مناسب مقطع راهنمایی دوم)"
        if score >= 60: return "استاندارد (مناسب مقطع دبیرستان)"
        if score >= 50: return "نسبتاً دشوار (مناسب سال اول دانشگاه)"
        if score >= 30: return "دشوار (مناسب دانشجویان)"
        return "خیلی دشوار (مناسب متخصصان و تحصیلات تکمیلی)"

    def technical_seo_audit(self, url: str) -> Dict:
        """بررسی فنی سئو سایت با قابلیت‌های پیشرفته و بررسی robots.txt"""
        if not url or not isinstance(url, str):
            return {"error": "URL نامعتبر است."}

        cache_key = f"tech_audit_{hashlib.md5(url.encode('utf-8')).hexdigest()}"
        cached = self.cache.get(cache_key)
        if cached:
            self.logger.info(f"گزارش فنی سئو برای URL '{url}' از کش بازگردانده شد.")
            return cached

        start_time = time.time()
        self.logger.info(f"شروع بررسی فنی سئو برای URL: {url}")

        try:
            # 1. تحلیل اولیه URL
            url_analysis = self.processor.analyze_url(url)
            if not url_analysis:
                return {"error": "URL نامعتبر یا غیرقابل تحلیل است."}
            if not url_analysis['is_https']:
                self.logger.warning(f"URL از HTTPS استفاده نمی‌کند: {url}")
                # می‌توان در اینجا تحلیل را متوقف کرد یا ادامه داد

            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # 2. بررسی robots.txt
            robots_url = urljoin(base_url, '/robots.txt')
            can_fetch = True
            try:
                self.robot_parser.set_url(robots_url)
                self.robot_parser.read()
                can_fetch = self.robot_parser.can_fetch(self.config.USER_AGENT, url)
                if not can_fetch:
                    self.logger.warning(f"دسترسی به URL '{url}' توسط robots.txt برای '{self.config.USER_AGENT}' ممنوع شده است.")
                    # تصمیم گیری: آیا متوقف شویم یا با احتیاط ادامه دهیم؟
                    # فعلا ادامه می‌دهیم اما در گزارش ذکر می‌کنیم.
                    # return {"error": f"Crawling URL is disallowed by {robots_url}"}
            except Exception as e:
                self.logger.warning(f"خطا در خواندن یا تجزیه robots.txt از '{robots_url}': {e}")
                # اگر robots.txt قابل خواندن نیست، معمولاً می‌توان ادامه داد (با فرض مجاز بودن)

            # 3. دریافت محتوای سایت
            html_content = None
            response = None
            page_stats = {}
            request_error = None
            if can_fetch: # فقط اگر مجاز بودیم، درخواست می‌دهیم
                try:
                    headers = {'User-Agent': self.config.USER_AGENT}
                    response = requests.get(
                        url,
                        headers=headers,
                        timeout=self.config.REQUEST_TIMEOUT,
                        allow_redirects=True,
                        verify=True # فعال کردن بررسی SSL Certificate
                    )
                    response.raise_for_status() # بررسی خطاهای HTTP (4xx, 5xx)
                    # حدس زدن انکودینگ
                    response.encoding = response.apparent_encoding
                    html_content = response.text
                    load_time = response.elapsed.total_seconds() # زمان پاسخ دقیق‌تر از requests

                    page_stats = {
                        "size_kb": round(len(html_content.encode('utf-8')) / 1024, 2), # اندازه بایت‌های واقعی
                        "load_time_sec": round(load_time, 3),
                        "status_code": response.status_code,
                        "redirect_count": len(response.history),
                        "final_url": response.url,
                        "content_type": response.headers.get('Content-Type', 'نامشخص'),
                        "server": response.headers.get('Server', 'نامشخص'),
                    }
                    self.logger.info(f"محتوای '{url}' با موفقیت دریافت شد. زمان بارگذاری: {load_time:.3f} ثانیه.")

                except Timeout:
                    request_error = "Timeout: زمان انتظار برای دریافت پاسخ از سرور به پایان رسید."
                    self.logger.error(f"{request_error} URL: {url}")
                except ConnectionError:
                    request_error = "ConnectionError: اتصال به سرور برقرار نشد یا مشکلی در شبکه وجود دارد."
                    self.logger.error(f"{request_error} URL: {url}")
                except HTTPError as e:
                    request_error = f"HTTP Error: {e.response.status_code} {e.response.reason}"
                    self.logger.error(f"{request_error} URL: {url}")
                except RequestException as e:
                    request_error = f"Request Error: خطای نامشخص در درخواست - {type(e).__name__}"
                    self.logger.error(f"{request_error} URL: {url} - {e}")
                except Exception as e:
                     request_error = f"Unexpected Error: خطای غیرمنتظره در حین درخواست - {type(e).__name__}"
                     self.logger.error(f"{request_error} URL: {url} - {e}")
            else:
                 request_error = f"Fetching disallowed by robots.txt at {robots_url}"


            # 4. استخراج عناصر سئو (فقط اگر محتوا دریافت شده باشد)
            seo_elements = {}
            if html_content:
                seo_elements = self.processor.extract_seo_elements(html_content, url)
            else:
                # اگر محتوا دریافت نشد، نمی‌توان عناصر را استخراج کرد
                 self.logger.warning(f"محتوایی برای استخراج عناصر سئو از {url} وجود ندارد.")


            # 5. تحلیل سرعت (بر اساس داده‌های موجود)
            # توجه: این تحلیل بسیار ساده است و جایگزین ابزارهای تخصصی نمی‌شود.
            speed_analysis = self._analyze_page_speed(html_content, response)

            # 6. ساخت گزارش فنی
            audit_report = {
                "audit_url": url,
                "audit_timestamp": datetime.now().isoformat(),
                "url_analysis": url_analysis,
                "robots_txt_check": {
                    "robots_url": robots_url,
                    "can_fetch": can_fetch,
                    "checked_user_agent": self.config.USER_AGENT,
                },
                "request_info": {
                     "success": html_content is not None and request_error is None,
                     "error_message": request_error,
                     "page_stats": page_stats,
                },
                "seo_elements": seo_elements, # ممکن است خالی باشد اگر درخواست ناموفق بود
                "speed_analysis": speed_analysis, # ممکن است محدود باشد اگر درخواست ناموفق بود
                "recommendations": "توصیه‌ها تولید نشدند (خطا در دریافت یا تحلیل صفحه)." # مقدار پیش‌فرض
            }

            # 7. تولید توصیه‌ها (فقط اگر تحلیل موفق بود)
            if audit_report["request_info"]["success"] and audit_report["seo_elements"]:
                 recommendations_prompt = self._create_technical_recommendation_prompt(audit_report)
                 self.logger.info(f"در حال تولید توصیه‌های فنی برای {url}...")
                 recommendations = self.generate_seo_advice(
                     recommendations_prompt,
                     max_length=400, # کمی طولانی‌تر برای توصیه‌های فنی
                     temperature=0.65
                 )
                 audit_report["recommendations"] = recommendations
            else:
                 self.logger.warning(f"تولید توصیه‌های فنی برای {url} به دلیل خطای قبلی لغو شد.")


            self.cache.set(cache_key, audit_report)
            total_time = time.time() - start_time
            self.logger.info(f"بررسی فنی سئو برای URL '{url}' کامل شد. زمان کل: {total_time:.2f} ثانیه.")
            return audit_report

        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"خطای غیرمنتظره کلی در بررسی فنی سایت '{url}': {e}. زمان کل: {total_time:.2f} ثانیه.", exc_info=True)
            return {"error": f"خطای غیرمنتظره کلی در بررسی فنی: {str(e)}"}


    def _create_technical_recommendation_prompt(self, audit_report: Dict) -> str:
         """ایجاد پرامپت برای تولید توصیه‌های فنی بر اساس گزارش"""
         url = audit_report.get("audit_url", "نامشخص")
         seo = audit_report.get("seo_elements", {})
         url_an = audit_report.get("url_analysis", {})
         speed = audit_report.get("speed_analysis", {})
         stats = audit_report.get("request_info", {}).get("page_stats", {})

         prompt = f"گزارش فنی سئو برای سایت {url}:\n"
         prompt += f"- وضعیت دسترسی robots.txt: {'مجاز' if audit_report.get('robots_txt_check', {}).get('can_fetch') else 'نامشخص یا غیرمجاز'}\n"
         prompt += f"- وضعیت درخواست HTTP: {'موفق' if audit_report.get('request_info', {}).get('success') else 'ناموفق'} ({audit_report.get('request_info', {}).get('error_message', '')})\n"
         prompt += f"- URL HTTPS: {'بله' if url_an.get('is_https') else 'خیر'}\n"
         prompt += f"- امتیاز سئو URL: {url_an.get('seo_friendly_score', 'N/A')}/100\n"
         prompt += f"- عنوان صفحه: {'دارد' if seo.get('title') else 'ندارد'} (طول: {seo.get('title_length', 0)})\n"
         prompt += f"- توضیحات متا: {'دارد' if seo.get('meta_description') else 'ندارد'} (طول: {seo.get('meta_description_length', 0)})\n"
         prompt += f"- تعداد H1: {seo.get('h1_count', 'N/A')}\n"
         prompt += f"- تصاویر بدون Alt: {seo.get('images', {}).get('without_alt', 'N/A')} / {seo.get('images', {}).get('count', 'N/A')}\n"
         prompt += f"- لینک Canonical: {'دارد' if seo.get('canonical_url') else 'ندارد'}\n"
         prompt += f"- Schema Markup: {'دارد' if seo.get('has_schema_markup') else 'ندارد'}\n"
         prompt += f"- Viewport Meta: {'دارد' if seo.get('has_viewport_meta') else 'ندارد'}\n"
         prompt += f"- حجم صفحه: {stats.get('size_kb', 'N/A')} KB\n"
         prompt += f"- زمان بارگذاری: {stats.get('load_time_sec', 'N/A')} ثانیه\n"
         prompt += f"- امتیاز سرعت (تخمینی): {speed.get('score', 'N/A')}/100\n"
         prompt += f"- مشکلات سرعت شناسایی شده: {', '.join(speed.get('issues', []) or ['هیچ'])} \n\n"
         prompt += "بر اساس این گزارش، مهم‌ترین توصیه‌های فنی برای بهبود سئوی این صفحه چیست؟ (به صورت لیستی و عملیاتی)"

         return prompt


    def _analyze_page_speed(self, html_content: Optional[str], response: Optional[requests.Response]) -> Dict:
        """تحلیل سرعت صفحه با معیارهای پایه (نیاز به ابزار دقیق‌تر دارد)"""
        # هشدار: این تحلیل بسیار ساده است و معیارهای Core Web Vitals را نمی‌سنجد.
        if not html_content or not response:
            return {"score": 0, "issues": ["محتوا یا پاسخ سرور برای تحلیل سرعت در دسترس نیست."], "details": {}}

        try:
            issues = []
            details = {}

            # 1. زمان پاسخ سرور (TTFB تقریبی)
            response_time = response.elapsed.total_seconds()
            details['server_response_time_sec'] = round(response_time, 3)
            if response_time > 0.8: # معیار سختگیرانه‌تر برای TTFB
                issues.append(f"زمان پاسخ اولیه سرور نسبتاً بالا است ({response_time:.3f} ثانیه). بهبود TTFB توصیه می‌شود.")
            elif response_time > 2.0:
                 issues.append(f"زمان پاسخ اولیه سرور بسیار بالا است ({response_time:.3f} ثانیه). نیاز به بررسی فوری دارد.")

            # 2. حجم صفحه
            page_size_kb = round(len(html_content.encode('utf-8')) / 1024, 2)
            details['page_size_kb'] = page_size_kb
            if page_size_kb > 1000: # ۱ مگابایت
                issues.append(f"حجم کل صفحه زیاد است ({page_size_kb} KB). بهینه‌سازی منابع (تصاویر، JS, CSS) ضروری است.")
            elif page_size_kb > 500:
                issues.append(f"حجم صفحه قابل قبول است اما می‌تواند کمتر شود ({page_size_kb} KB).")

            # 3. تعداد درخواست‌ها (تخمینی از روی لینک‌ها و اسکریپت‌ها)
            soup = BeautifulSoup(html_content, 'html.parser')
            css_links = len(soup.find_all('link', rel='stylesheet', href=True))
            js_scripts = len(soup.find_all('script', src=True))
            img_tags = len(soup.find_all('img', src=True))
            # این فقط تعداد تگ‌هاست، نه درخواست‌های واقعی شبکه
            total_resources = css_links + js_scripts + img_tags
            details['resource_counts'] = {'css': css_links, 'js': js_scripts, 'img': img_tags, 'total_tags': total_resources}
            if total_resources > 80:
                issues.append(f"تعداد تخمینی منابع خارجی (CSS, JS, IMG tags) زیاد است ({total_resources}). کاهش درخواست‌ها یا ترکیب فایل‌ها توصیه می‌شود.")
            elif total_resources > 50:
                 issues.append(f"تعداد تخمینی منابع خارجی (CSS, JS, IMG tags) نسبتاً زیاد است ({total_resources}).")


            # 4. فشرده‌سازی (بررسی سرآیند Content-Encoding)
            encoding = response.headers.get('Content-Encoding')
            details['content_encoding'] = encoding
            if encoding not in ['gzip', 'br', 'deflate']:
                 issues.append("فشرده‌سازی Gzip یا Brotli برای محتوای متنی فعال نشده است.")


            # محاسبه امتیاز ساده (بسیار تقریبی)
            score = 100
            score -= min(max(0, (response_time - 0.5) * 30), 30) # جریمه برای TTFB بالا
            score -= min(max(0, (page_size_kb - 300) / 50), 30) # جریمه برای حجم بالا
            score -= min(max(0, (total_resources - 40) / 5), 20) # جریمه برای منابع زیاد
            if encoding not in ['gzip', 'br', 'deflate']: score -= 10 # جریمه برای عدم فشرده‌سازی
            score = max(0, round(score))

            return {
                "score": score,
                "issues": issues,
                "details": details
            }
        except Exception as e:
            self.logger.warning(f"خطا در تحلیل ساده سرعت صفحه: {e}", exc_info=True)
            return {"score": 0, "issues": [f"خطا در تحلیل سرعت: {e}"], "details": {}}

    # --- Placeholder Methods ---
    # این متدها نیاز به پیاده‌سازی واقعی دارند (مثلاً با API های خارجی یا دیتابیس)

    def research_keywords(self, topic: str) -> Dict:
        """تحقیق کلمات کلیدی (Placeholder)."""
        self.logger.warning("متد research_keywords یک Placeholder است و نیاز به پیاده‌سازی واقعی دارد.")
        prompt = f"در مورد تحقیق کلمات کلیدی برای موضوع '{topic}' چه نکاتی مهم است و چه ابزارهایی پیشنهاد می‌کنید؟"
        advice = self.generate_seo_advice(prompt, max_length=300)
        return {
            "topic": topic,
            "status": "Placeholder - Requires Real Implementation",
            "placeholder_advice": advice,
            "timestamp": datetime.now().isoformat()
        }

    def analyze_competitor(self, competitor_url: str) -> Dict:
        """تحلیل رقیب (Placeholder - از تحلیل فنی استفاده می‌کند)."""
        self.logger.warning("متد analyze_competitor یک Placeholder است و از technical_seo_audit استفاده می‌کند.")
        # می‌توان تحلیل فنی رقیب را انجام داد
        audit_result = self.technical_seo_audit(competitor_url)
        # می‌توان نتایج را خلاصه کرد یا مقایسه انجام داد (نیاز به پیاده‌سازی دارد)
        return {
            "competitor_url": competitor_url,
            "status": "Placeholder - Using Technical Audit",
            "technical_audit_summary": {
                "success": audit_result.get("request_info", {}).get("success", False),
                "error": audit_result.get("request_info", {}).get("error_message"),
                "title": audit_result.get("seo_elements", {}).get("title"),
                "h1_count": audit_result.get("seo_elements", {}).get("h1_count"),
                "load_time": audit_result.get("request_info", {}).get("page_stats", {}).get("load_time_sec"),
            },
            "timestamp": datetime.now().isoformat()
            # "full_audit_data": audit_result # (اختیاری)
        }

    def analyze_trends(self, topic: str) -> Dict:
        """تحلیل ترندهای سئو (Placeholder)."""
        self.logger.warning("متد analyze_trends یک Placeholder است و نیاز به پیاده‌سازی واقعی دارد.")
        prompt = f"چگونه می‌توانم ترندهای سئو مرتبط با موضوع '{topic}' را شناسایی و از آنها استفاده کنم؟"
        advice = self.generate_seo_advice(prompt, max_length=300)
        return {
            "topic": topic,
            "status": "Placeholder - Requires Real Implementation (e.g., Google Trends API)",
            "placeholder_advice": advice,
            "timestamp": datetime.now().isoformat()
        }

    # --- ذخیره گزارش ---
    def save_report(self, report_data: Dict, report_type: str = "general", format: str = "json") -> Optional[str]:
        """ذخیره گزارش در فرمت‌های مختلف"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # استخراج نام فایل از URL یا استفاده از نوع گزارش
            filename_base = "seo_report"
            if 'url' in report_data:
                 try:
                      domain = urlparse(report_data['url']).netloc
                      filename_base = domain.replace('.', '_') + f"_{report_type}"
                 except:
                      filename_base = f"report_{report_type}" # Fallback
            elif 'audit_url' in report_data:
                 try:
                      domain = urlparse(report_data['audit_url']).netloc
                      filename_base = domain.replace('.', '_') + f"_{report_type}"
                 except:
                      filename_base = f"report_{report_type}" # Fallback
            else:
                 filename_base = f"report_{report_type}"


            filename = Path(self.config.REPORTS_DIR) / f"{filename_base}_{timestamp}"

            if format == "json":
                filepath = filename.with_suffix(".json")
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, ensure_ascii=False, indent=4, default=str) # default=str for datetime etc.
            elif format == "txt":
                filepath = filename.with_suffix(".txt")
                with open(filepath, 'w', encoding='utf-8') as f:
                    self._write_text_report(f, report_data, report_type)
            elif format == "md":
                filepath = filename.with_suffix(".md")
                with open(filepath, 'w', encoding='utf-8') as f:
                    self._write_markdown_report(f, report_data, report_type)
            else:
                self.logger.error(f"فرمت گزارش نامعتبر: {format}. فرمت‌های مجاز: json, txt, md")
                return None

            self.logger.info(f"گزارش '{report_type}' با موفقیت در فایل '{filepath}' ذخیره شد.")
            return str(filepath)
        except Exception as e:
            self.logger.error(f"خطا در ذخیره گزارش '{report_type}': {e}", exc_info=True)
            return None

    def _write_text_report(self, file, report_data: Dict, report_type: str):
        """نوشتن گزارش متنی (بهبود یافته)"""
        title = f"گزارش سئو - نوع: {report_type.replace('_', ' ').title()}"
        file.write(title + "\n")
        file.write("=" * len(title) + "\n")
        file.write(f"تاریخ تولید: {report_data.get('timestamp') or report_data.get('audit_timestamp') or datetime.now().isoformat()}\n")
        url = report_data.get('url') or report_data.get('audit_url') or report_data.get('source')
        if url: file.write(f"منبع/URL: {url}\n")
        file.write("-" * len(title) + "\n\n")

        if report_type == "technical_audit":
            req_info = report_data.get("request_info", {})
            file.write(f"وضعیت درخواست: {'موفق' if req_info.get('success') else 'ناموفق'}\n")
            if not req_info.get('success'): file.write(f"  پیام خطا: {req_info.get('error_message')}\n")
            file.write(f"وضعیت robots.txt: {'مجاز' if report_data.get('robots_txt_check', {}).get('can_fetch') else 'نامشخص یا غیرمجاز'}\n\n")

            if req_info.get('success'):
                stats = req_info.get('page_stats', {})
                file.write("آمار صفحه:\n")
                file.write(f"  کد وضعیت: {stats.get('status_code', 'N/A')}\n")
                file.write(f"  زمان بارگذاری: {stats.get('load_time_sec', 'N/A')} ثانیه\n")
                file.write(f"  حجم صفحه: {stats.get('size_kb', 'N/A')} KB\n")
                file.write(f"  URL نهایی: {stats.get('final_url', 'N/A')}\n\n")

                seo = report_data.get("seo_elements", {})
                file.write("عناصر کلیدی سئو:\n")
                file.write(f"  عنوان: {seo.get('title', 'ندارد')}\n")
                file.write(f"  توضیحات متا: {seo.get('meta_description', 'ندارد')}\n")
                file.write(f"  تعداد H1: {seo.get('h1_count', 'N/A')}\n")
                img = seo.get('images', {})
                file.write(f"  تصاویر: {img.get('count', 'N/A')} (بدون Alt: {img.get('without_alt', 'N/A')})\n")
                file.write(f"  لینک Canonical: {seo.get('canonical_url', 'ندارد')}\n")
                file.write(f"  Schema Markup: {'دارد' if seo.get('has_schema_markup') else 'ندارد'}\n")
                file.write(f"  Viewport Meta: {'دارد' if seo.get('has_viewport_meta') else 'ندارد'}\n\n")

                speed = report_data.get("speed_analysis", {})
                file.write("تحلیل سرعت (تخمینی):\n")
                file.write(f"  امتیاز: {speed.get('score', 'N/A')}/100\n")
                file.write("  مشکلات شناسایی شده:\n")
                for issue in speed.get('issues', []): file.write(f"    - {issue}\n")
                if not speed.get('issues'): file.write("    - موردی یافت نشد.\n")
                file.write("\n")

        elif report_type == "content_analysis":
            metrics = report_data.get('metrics', {})
            file.write("معیارهای محتوا:\n")
            file.write(f"  تعداد کلمات: {metrics.get('word_count', 'N/A')}\n")
            file.write(f"  امتیاز خوانایی: {metrics.get('readability_score', 'N/A')}/100 ({metrics.get('readability_level', 'N/A')})\n")
            file.write(f"  احساس کلی: {metrics.get('sentiment', 'N/A')}\n\n")

            keywords = report_data.get('keyword_analysis', {})
            file.write("کلمات کلیدی اصلی (Top 10):\n")
            if keywords:
                 # مرتب سازی مجدد برای اطمینان
                 sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
                 for i, (word, density) in enumerate(sorted_keywords[:10]):
                      file.write(f"  {i+1}. {word}: {density:.2f}%\n")
            else:
                 file.write("  کلمه کلیدی یافت نشد.\n")
            file.write("\n")

        # نوشتن توصیه‌ها برای هر دو نوع گزارش (اگر وجود داشته باشد)
        recommendations = report_data.get('recommendations')
        if recommendations:
             file.write("توصیه‌ها:\n")
             file.write(recommendations)
        else:
             file.write("توصیه‌ها: تولید نشد یا در دسترس نیست.\n")

    def _write_markdown_report(self, file, report_data: Dict, report_type: str):
        """نوشتن گزارش Markdown (بهبود یافته)"""
        title = f"گزارش سئو - {report_type.replace('_', ' ').title()}"
        file.write(f"# {title}\n\n")
        ts = report_data.get('timestamp') or report_data.get('audit_timestamp') or datetime.now().isoformat()
        file.write(f"**تاریخ تولید:** `{ts}`\n")
        url = report_data.get('url') or report_data.get('audit_url') or report_data.get('source')
        if url: file.write(f"**منبع/URL:** {url}\n\n")
        file.write("---\n\n")

        if report_type == "technical_audit":
            req_info = report_data.get("request_info", {})
            file.write("## وضعیت کلی\n")
            file.write(f"- **وضعیت درخواست:** {'✅ موفق' if req_info.get('success') else '❌ ناموفق'}\n")
            if not req_info.get('success'): file.write(f"  - **پیام خطا:** `{req_info.get('error_message')}`\n")
            can_fetch = report_data.get('robots_txt_check', {}).get('can_fetch')
            file.write(f"- **وضعیت robots.txt:** {'✅ مجاز' if can_fetch else ('❌ غیرمجاز' if can_fetch is False else '⚠️ نامشخص')}\n\n")

            if req_info.get('success'):
                stats = req_info.get('page_stats', {})
                file.write("## آمار صفحه\n")
                file.write(f"- **کد وضعیت:** `{stats.get('status_code', 'N/A')}`\n")
                file.write(f"- **زمان بارگذاری:** `{stats.get('load_time_sec', 'N/A')} ثانیه`\n")
                file.write(f"- **حجم صفحه:** `{stats.get('size_kb', 'N/A')} KB`\n")
                file.write(f"- **URL نهایی:** `{stats.get('final_url', 'N/A')}`\n\n")

                seo = report_data.get("seo_elements", {})
                file.write("## عناصر کلیدی سئو\n")
                file.write(f"- **عنوان:** `{seo.get('title', 'ندارد')}` (طول: {seo.get('title_length', 0)})\n")
                file.write(f"- **توضیحات متا:** `{seo.get('meta_description', 'ندارد')}` (طول: {seo.get('meta_description_length', 0)})\n")
                file.write(f"- **تعداد H1:** `{seo.get('h1_count', 'N/A')}`\n")
                img = seo.get('images', {})
                file.write(f"- **تصاویر:** {img.get('count', 'N/A')} (بدون Alt: **{img.get('without_alt', 'N/A')}** - {img.get('missing_alt_ratio', 0):.1f}%)\n")
                file.write(f"- **لینک Canonical:** `{seo.get('canonical_url', 'ندارد')}`\n")
                file.write(f"- **Schema Markup:** {'✅ دارد' if seo.get('has_schema_markup') else '❌ ندارد'}\n")
                file.write(f"- **Viewport Meta:** {'✅ دارد' if seo.get('has_viewport_meta') else '❌ ندارد'}\n\n")

                speed = report_data.get("speed_analysis", {})
                file.write("## تحلیل سرعت (تخمینی)\n")
                file.write(f"- **امتیاز:** `{speed.get('score', 'N/A')}/100`\n")
                file.write("**مشکلات شناسایی شده:**\n")
                if speed.get('issues'):
                    for issue in speed.get('issues', []): file.write(f"  - ⚠️ {issue}\n")
                else:
                    file.write("  - ✅ موردی یافت نشد.\n")
                file.write("\n")

        elif report_type == "content_analysis":
            metrics = report_data.get('metrics', {})
            file.write("## معیارهای محتوا\n")
            file.write(f"- **تعداد کلمات:** `{metrics.get('word_count', 'N/A')}`\n")
            readability_score = metrics.get('readability_score', 'N/A')
            readability_level = metrics.get('readability_level', 'N/A')
            file.write(f"- **امتیاز خوانایی:** `{readability_score}/100` ({readability_level})\n")
            file.write(f"- **احساس کلی:** `{metrics.get('sentiment', 'N/A')}`\n\n")

            keywords = report_data.get('keyword_analysis', {})
            file.write("## کلمات کلیدی اصلی (Top 10)\n")
            if keywords:
                 file.write("| رتبه | کلمه کلیدی | تراکم (%) |\n")
                 file.write("|---|---|---|\n")
                 # مرتب سازی مجدد برای اطمینان
                 sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
                 for i, (word, density) in enumerate(sorted_keywords[:10]):
                      file.write(f"| {i+1} | {word} | {density:.2f} |\n")
            else:
                 file.write("کلمه کلیدی یافت نشد.\n")
            file.write("\n")

        # نوشتن توصیه‌ها
        recommendations = report_data.get('recommendations')
        file.write("## توصیه‌ها\n")
        if recommendations and recommendations != "توصیه‌ها تولید نشدند (خطا در دریافت یا تحلیل صفحه).":
            file.write("```text\n")
            file.write(recommendations + "\n")
            file.write("```\n")
        else:
            file.write("توصیه‌ها تولید نشد یا در دسترس نیست.\n")


# --- رابط کاربری خط فرمان ---
class SEOExpertUI:
    def __init__(self, model: SEOExpertModel):
        self.model = model
        self.config = model.config
        self.logger = logging.getLogger(f"{logger.name}.SEOExpertUI")
        self._setup_menu()
        self.logger.info("رابط کاربری مقداردهی اولیه شد.")

    def _setup_menu(self):
        """تنظیم منوی تعاملی"""
        self.menu_options = {
            "1": {
                "title": "مشاوره سئو",
                "action": self.seo_consultation,
                "description": "پرسش سوالات تخصصی سئو و دریافت پاسخ"
            },
            "2": {
                "title": "تحلیل محتوا",
                "action": self.content_analysis,
                "description": "تحلیل محتوای متنی یا یک URL از نظر سئو"
            },
            "3": {
                "title": "بررسی فنی سایت",
                "action": self.technical_audit,
                "description": "بررسی فنی یک URL از نظر معیارهای سئو"
            },
            "4": {
                "title": "تحلیل کلمات کلیدی متن/URL",
                "action": self.keyword_analysis_ui,
                "description": "نمایش کلمات کلیدی اصلی و تراکم آنها از متن یا URL"
            },
            "5": {
                "title": "گزارش کامل سئو URL",
                "action": self.full_seo_report,
                "description": "تهیه گزارش ترکیبی فنی و محتوا برای یک URL"
            },
            "6": {
                "title": "تنظیمات و ابزارها",
                "action": self.settings_menu,
                "description": "مشاهده تنظیمات و اجرای ابزارهای دیگر"
            },
            "7": {
                "title": "خروج",
                "action": self.exit_app,
                "description": "خروج از برنامه"
            }
        }

    def display_menu(self):
        """نمایش منوی اصلی"""
        print(Style.BRIGHT + Fore.MAGENTA + "\n" + "=" * 60)
        print(Fore.YELLOW + "مدل هوش مصنوعی متخصص سئو".center(70)) # Center alignment adjusted
        print(Fore.CYAN + f"مدل پایه: {self.config.MODEL_NAME}".center(70))
        print(Fore.GREEN + f"دستگاه: {self.config.DEVICE.upper()}".center(70))
        print(Fore.MAGENTA + "=" * 60 + Style.RESET_ALL)

        for key, option in self.menu_options.items():
            print(f"{Fore.BLUE}{key}.{Style.BRIGHT} {option['title']}{Style.RESET_ALL} - {Fore.LIGHTBLACK_EX}{option['description']}{Fore.RESET}")

        print(Fore.MAGENTA + "=" * 60 + Style.RESET_ALL)

    def run(self):
        """اجرای حلقه اصلی رابط کاربری"""
        try:
            while True:
                self.display_menu()
                choice = input(Fore.WHITE + Style.BRIGHT + "لطفاً گزینه مورد نظر را انتخاب کنید (1-7): " + Style.RESET_ALL).strip()

                if choice in self.menu_options:
                    action = self.menu_options[choice]["action"]
                    self.logger.info(f"کاربر گزینه '{self.menu_options[choice]['title']}' ({choice}) را انتخاب کرد.")
                    try:
                        action() # اجرای متد مربوط به گزینه
                    except Exception as e:
                        self.logger.error(f"خطا در اجرای اکشن برای گزینه {choice}: {e}", exc_info=True)
                        print(Fore.RED + Style.BRIGHT + f"\nخطا در پردازش درخواست: {str(e)}" + Style.RESET_ALL)
                elif choice == '7':
                     self.exit_app() # خروج مستقیم اگر کاربر ۷ را وارد کرد
                     break
                else:
                    print(Fore.RED + "\nگزینه نامعتبر! لطفاً عددی بین 1 تا 7 وارد کنید." + Fore.RESET)

                # توقف کوتاه برای خواندن خروجی قبل از نمایش مجدد منو
                if choice != '7':
                    input(Fore.LIGHTBLACK_EX + "\nبرای بازگشت به منو Enter بزنید..." + Fore.RESET)

        except KeyboardInterrupt:
            self.logger.info("خروج توسط کاربر (Ctrl+C).")
            self.exit_app()
        except Exception as e:
            self.logger.critical(f"خطای غیرمنتظره در حلقه اصلی رابط کاربری: {e}", exc_info=True)
            print(Fore.RED + Style.BRIGHT + f"\nخطای بحرانی رخ داد: {e}. برنامه خاتمه می‌یابد." + Style.RESET_ALL)
            self.exit_app()

    def get_valid_input(self, prompt: str, min_length: int = 1) -> str:
         """دریافت ورودی معتبر از کاربر"""
         while True:
              user_input = input(prompt).strip()
              if len(user_input) >= min_length:
                   return user_input
              else:
                   print(Fore.RED + f"ورودی نامعتبر. لطفاً حداقل {min_length} کاراکتر وارد کنید." + Fore.RESET)


    def get_valid_url(self, prompt: str) -> Optional[str]:
         """دریافت و اعتبارسنجی اولیه URL از کاربر"""
         while True:
              url = input(prompt).strip()
              if not url:
                   print(Fore.YELLOW + "ورودی خالی است. لطفاً URL را وارد کنید یا برای لغو 'c' را بزنید." + Fore.RESET)
                   continue
              if url.lower() == 'c':
                  return None

              parsed = urlparse(url)
              if parsed.scheme in ['http', 'https'] and parsed.netloc:
                   return url
              else:
                   print(Fore.RED + "فرمت URL نامعتبر به نظر می‌رسد. لطفاً URL کامل (شامل http:// یا https://) وارد کنید." + Fore.RESET)


    def ask_to_save_report(self, report_data: Dict, report_type: str) -> None:
         """پرسش از کاربر برای ذخیره گزارش و انجام آن"""
         save_choice = input(Fore.CYAN + "\nآیا مایل به ذخیره این گزارش هستید؟ (y/n/m: markdown): " + Fore.RESET).lower().strip()
         if save_choice == 'y':
              format = "json"
              filepath = self.model.save_report(report_data, report_type, format=format)
              if filepath:
                   print(Fore.GREEN + f"\nگزارش با فرمت JSON در فایل '{filepath}' ذخیره شد." + Fore.RESET)
              else:
                   print(Fore.RED + "خطا در ذخیره گزارش JSON." + Fore.RESET)
         elif save_choice == 'm':
               format = "md"
               filepath = self.model.save_report(report_data, report_type, format=format)
               if filepath:
                    print(Fore.GREEN + f"\nگزارش با فرمت Markdown در فایل '{filepath}' ذخیره شد." + Fore.RESET)
               else:
                    print(Fore.RED + "خطا در ذخیره گزارش Markdown." + Fore.RESET)

    # --- پیاده‌سازی اکشن‌های منو ---

    def seo_consultation(self):
        """1. مشاوره سئو"""
        print(Fore.CYAN + Style.BRIGHT + "\n=== مشاوره تخصصی سئو ===" + Style.RESET_ALL)
        question = self.get_valid_input(Fore.WHITE + "سوال سئو خود را مطرح کنید: " + Fore.RESET, min_length=10)

        print(Fore.YELLOW + "\nدر حال پردازش سوال و تولید پاسخ... لطفاً منتظر بمانید." + Style.RESET_ALL)
        start_time = time.time()
        response = self.model.generate_seo_advice(question)
        end_time = time.time()
        print(f"{Fore.LIGHTBLACK_EX}(زمان پردازش: {end_time - start_time:.2f} ثانیه){Fore.RESET}")

        print(Fore.GREEN + Style.BRIGHT + "\nپاسخ متخصص سئو:" + Style.RESET_ALL)
        print(Fore.WHITE + response + Fore.RESET)

        # ذخیره پاسخ
        report_data = {
            "type": "seo_consultation",
            "question": question,
            "answer": response,
            "timestamp": datetime.now().isoformat()
        }
        self.ask_to_save_report(report_data, "consultation")

    def content_analysis(self):
        """2. تحلیل محتوا"""
        print(Fore.CYAN + Style.BRIGHT + "\n=== تحلیل محتوای سئو ===" + Style.RESET_ALL)
        choice = input(Fore.WHITE + "تحلیل بر اساس 'URL' یا 'Text'? (u/t): " + Fore.RESET).lower().strip()

        report_data = None
        start_time = time.time()

        if choice == 'u':
            url = self.get_valid_url(Fore.WHITE + "آدرس URL صفحه مورد نظر را وارد کنید: " + Fore.RESET)
            if url:
                print(Fore.YELLOW + f"\nدر حال دریافت و تحلیل محتوای URL: {url}..." + Style.RESET_ALL)
                report_data = self.model.analyze_content(url=url)
        elif choice == 't':
            print(Fore.WHITE + "متن مورد نظر را وارد کنید (برای پایان، یک خط خالی و سپس Enter بزنید):" + Fore.RESET)
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                         break
                    lines.append(line)
                except EOFError: # Handle Ctrl+D
                    break
            text = "\n".join(lines)
            if text:
                print(Fore.YELLOW + "\nدر حال تحلیل متن وارد شده..." + Style.RESET_ALL)
                report_data = self.model.analyze_content(text=text)
            else:
                 print(Fore.RED + "متنی برای تحلیل وارد نشد." + Fore.RESET)
        else:
            print(Fore.RED + "انتخاب نامعتبر." + Fore.RESET)
            return

        end_time = time.time()
        print(f"{Fore.LIGHTBLACK_EX}(زمان پردازش: {end_time - start_time:.2f} ثانیه){Fore.RESET}")

        if report_data:
            if "error" in report_data:
                print(Fore.RED + Style.BRIGHT + f"\nخطا در تحلیل: {report_data['error']}" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + Style.BRIGHT + "\n--- نتایج تحلیل محتوا ---" + Style.RESET_ALL)
                metrics = report_data.get('metrics', {})
                print(f"منبع: {report_data.get('source', 'نامشخص')}")
                print(f"تعداد کلمات: {metrics.get('word_count', 'N/A')}")
                print(f"امتیاز خوانایی: {metrics.get('readability_score', 'N/A')}/100 ({metrics.get('readability_level', 'N/A')})")
                print(f"احساس کلی: {metrics.get('sentiment', 'N/A')}")

                print(Fore.YELLOW + "\nکلمات کلیدی اصلی (Top 5):" + Fore.RESET)
                keywords = report_data.get('keyword_analysis', {})
                if keywords:
                     sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
                     for i, (word, density) in enumerate(sorted_keywords[:5]):
                          print(f"  {i+1}. {word}: {density:.2f}%")
                else:
                     print("  کلمه کلیدی یافت نشد.")

                print(Fore.YELLOW + "\nتوصیه‌ها:" + Fore.RESET)
                print(Fore.WHITE + report_data.get('recommendations', 'توصیه‌ای تولید نشد.') + Fore.RESET)

                self.ask_to_save_report(report_data, "content_analysis")
        else:
             # خطا قبلاً چاپ شده است
             pass

    def technical_audit(self):
        """3. بررسی فنی سایت"""
        print(Fore.CYAN + Style.BRIGHT + "\n=== بررسی فنی سئو سایت ===" + Style.RESET_ALL)
        url = self.get_valid_url(Fore.WHITE + "آدرس URL صفحه مورد نظر را وارد کنید: " + Fore.RESET)
        if not url: return # کاربر لغو کرد

        print(Fore.YELLOW + f"\nدر حال انجام بررسی فنی برای URL: {url}..." + Style.RESET_ALL)
        start_time = time.time()
        report_data = self.model.technical_seo_audit(url)
        end_time = time.time()
        print(f"{Fore.LIGHTBLACK_EX}(زمان پردازش: {end_time - start_time:.2f} ثانیه){Fore.RESET}")

        if "error" in report_data:
            print(Fore.RED + Style.BRIGHT + f"\nخطا در بررسی فنی: {report_data['error']}" + Style.RESET_ALL)
        else:
            print(Fore.GREEN + Style.BRIGHT + "\n--- نتایج بررسی فنی سئو ---" + Style.RESET_ALL)
            req_info = report_data.get("request_info", {})
            print(f"URL بررسی شده: {report_data.get('audit_url', 'N/A')}")
            print(f"وضعیت درخواست: {'✅ موفق' if req_info.get('success') else '❌ ناموفق'}")
            if not req_info.get('success'): print(f"  پیام خطا: {req_info.get('error_message')}")

            if req_info.get('success'):
                 stats = req_info.get('page_stats', {})
                 seo = report_data.get('seo_elements', {})
                 speed = report_data.get('speed_analysis', {})

                 print("\nخلاصه وضعیت:")
                 print(f"  زمان بارگذاری: {stats.get('load_time_sec', 'N/A')} ثانیه")
                 print(f"  حجم صفحه: {stats.get('size_kb', 'N/A')} KB")
                 print(f"  امتیاز سرعت (تخمینی): {speed.get('score', 'N/A')}/100")
                 print(f"  عنوان: {'دارد' if seo.get('title') else 'ندارد'}")
                 print(f"  توضیحات متا: {'دارد' if seo.get('meta_description') else 'ندارد'}")
                 print(f"  تعداد H1: {seo.get('h1_count', 'N/A')}")
                 img_alt = seo.get('images', {}).get('without_alt', 'N/A')
                 print(f"  تصاویر بدون Alt: {img_alt}")
                 print(f"  لینک Canonical: {'دارد' if seo.get('canonical_url') else 'ندارد'}")
                 print(f"  Schema Markup: {'دارد' if seo.get('has_schema_markup') else 'ندارد'}")
                 print(f"  Viewport Meta: {'دارد' if seo.get('has_viewport_meta') else 'ندارد'}")

                 print(Fore.YELLOW + "\nمشکلات سرعت شناسایی شده:" + Fore.RESET)
                 if speed.get('issues'):
                      for issue in speed.get('issues'): print(f"  - ⚠️ {issue}")
                 else:
                      print("  - ✅ موردی یافت نشد.")

                 print(Fore.YELLOW + "\nتوصیه‌ها:" + Fore.RESET)
                 print(Fore.WHITE + report_data.get('recommendations', 'توصیه‌ای تولید نشد.') + Fore.RESET)

                 self.ask_to_save_report(report_data, "technical_audit")
            else:
                 # اگر درخواست ناموفق بود، فقط پیام خطا نمایش داده می‌شود
                 pass

    def keyword_analysis_ui(self):
        """4. تحلیل کلمات کلیدی متن یا URL"""
        print(Fore.CYAN + Style.BRIGHT + "\n=== تحلیل کلمات کلیدی ===" + Style.RESET_ALL)
        # از تابع تحلیل محتوا استفاده می‌کنیم چون کلمات کلیدی را محاسبه می‌کند
        choice = input(Fore.WHITE + "تحلیل بر اساس 'URL' یا 'Text'? (u/t): " + Fore.RESET).lower().strip()

        report_data = None
        start_time = time.time()

        if choice == 'u':
            url = self.get_valid_url(Fore.WHITE + "آدرس URL صفحه مورد نظر را وارد کنید: " + Fore.RESET)
            if url:
                print(Fore.YELLOW + f"\nدر حال دریافت و تحلیل کلمات کلیدی URL: {url}..." + Style.RESET_ALL)
                # فقط برای کلمات کلیدی، تحلیل کامل محتوا را انجام می‌دهیم
                report_data = self.model.analyze_content(url=url)
        elif choice == 't':
            print(Fore.WHITE + "متن مورد نظر را وارد کنید (برای پایان، یک خط خالی و سپس Enter بزنید):" + Fore.RESET)
            lines = []
            while True:
                 try:
                     line = input()
                     if line == "": break
                     lines.append(line)
                 except EOFError: break
            text = "\n".join(lines)
            if text:
                print(Fore.YELLOW + "\nدر حال تحلیل کلمات کلیدی متن..." + Style.RESET_ALL)
                report_data = self.model.analyze_content(text=text)
            else:
                 print(Fore.RED + "متنی برای تحلیل وارد نشد." + Fore.RESET)
        else:
            print(Fore.RED + "انتخاب نامعتبر." + Fore.RESET)
            return

        end_time = time.time()
        print(f"{Fore.LIGHTBLACK_EX}(زمان پردازش: {end_time - start_time:.2f} ثانیه){Fore.RESET}")

        if report_data:
            if "error" in report_data:
                print(Fore.RED + Style.BRIGHT + f"\nخطا در تحلیل: {report_data['error']}" + Style.RESET_ALL)
            else:
                print(Fore.GREEN + Style.BRIGHT + "\n--- کلمات کلیدی اصلی و تراکم ---" + Style.RESET_ALL)
                keywords = report_data.get('keyword_analysis', {})
                if keywords:
                     sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)
                     print("| رتبه | کلمه کلیدی | تراکم (%) |")
                     print("|---|---|---|")
                     for i, (word, density) in enumerate(sorted_keywords[:20]): # نمایش ۲۰ کلمه برتر
                          print(f"| {i+1} | {word} | {density:.2f} |")
                     print("\n" + Fore.LIGHTBLACK_EX + "توجه: تراکم بر اساس فراوانی کلمات (پس از حذف استاپ وردها) محاسبه شده است." + Fore.RESET)
                else:
                     print("کلمه کلیدی قابل توجهی یافت نشد.")

                # نیازی به ذخیره مجدد نیست چون analyze_content این امکان را می‌دهد
                # self.ask_to_save_report(report_data, "keyword_analysis")
        else:
             # خطا قبلاً چاپ شده
             pass


    def full_seo_report(self):
        """5. گزارش کامل سئو برای URL"""
        print(Fore.CYAN + Style.BRIGHT + "\n=== گزارش کامل سئو برای URL ===" + Style.RESET_ALL)
        url = self.get_valid_url(Fore.WHITE + "آدرس URL صفحه مورد نظر را وارد کنید: " + Fore.RESET)
        if not url: return

        print(Fore.YELLOW + f"\nدر حال تهیه گزارش کامل برای URL: {url} (ممکن است کمی طول بکشد)..." + Style.RESET_ALL)
        start_time = time.time()

        # 1. انجام بررسی فنی
        print("مرحله 1: بررسی فنی...")
        tech_report = self.model.technical_seo_audit(url)
        if "error" in tech_report:
             print(Fore.RED + Style.BRIGHT + f"\nخطا در بخش فنی: {tech_report['error']}. گزارش کامل لغو شد." + Style.RESET_ALL)
             return

        # 2. انجام تحلیل محتوا (اگر بخش فنی موفق بود)
        content_report = None
        if tech_report.get("request_info", {}).get("success"):
             print("مرحله 2: تحلیل محتوا...")
             # از محتوای دانلود شده در بخش فنی استفاده نمی‌کنیم تا کش مجزا کار کند
             content_report = self.model.analyze_content(url=url)
             if content_report and "error" in content_report:
                  print(Fore.YELLOW + f"هشدار: خطا در بخش تحلیل محتوا: {content_report['error']}. گزارش فنی همچنان موجود است." + Fore.RESET)
                  content_report = None # ادامه با گزارش فنی
        else:
             print(Fore.YELLOW + "مرحله 2: تحلیل محتوا به دلیل عدم موفقیت دریافت صفحه رد شد." + Fore.RESET)


        # 3. ترکیب گزارش‌ها
        full_report = {
            "report_type": "full_seo_audit",
            "audit_url": url,
            "audit_timestamp": datetime.now().isoformat(),
            "technical_audit": tech_report,
            "content_analysis": content_report if content_report else {"status": "Skipped or Failed"}
        }

        end_time = time.time()
        print(f"\nگزارش کامل آماده شد. (زمان کل: {end_time - start_time:.2f} ثانیه)")

        # 4. نمایش خلاصه (اختیاری)
        print("\nخلاصه گزارش:")
        print(f"- وضعیت فنی: {'✅ موفق' if tech_report.get('request_info', {}).get('success') else '❌ ناموفق'}")
        if content_report:
            print(f"- وضعیت محتوا: {'✅ موفق' if 'error' not in content_report else '❌ ناموفق'}")
            print(f"- امتیاز خوانایی: {content_report.get('metrics', {}).get('readability_score', 'N/A')}")
            print(f"- تعداد کلمات: {content_report.get('metrics', {}).get('word_count', 'N/A')}")
        else:
            print("- وضعیت محتوا: ❌ ناموفق یا رد شده")


        # 5. ذخیره گزارش کامل
        print("\nگزارش کامل شامل جزئیات فنی و محتوا (در صورت موفقیت) است.")
        self.ask_to_save_report(full_report, "full_audit")


    def settings_menu(self):
        """6. منوی تنظیمات و ابزارها"""
        while True:
            print(Fore.CYAN + Style.BRIGHT + "\n=== تنظیمات و ابزارها ===" + Style.RESET_ALL)
            print("1. نمایش تنظیمات فعلی")
            print("2. پاک کردن کش")
            print("3. تحقیق کلمات کلیدی (Placeholder)")
            print("4. تحلیل رقیب (Placeholder)")
            print("5. تحلیل ترندها (Placeholder)")
            print("0. بازگشت به منوی اصلی")
            print(Fore.MAGENTA + "=" * 30 + Style.RESET_ALL)

            choice = input(Fore.WHITE + "گزینه مورد نظر را انتخاب کنید: " + Fore.RESET).strip()

            if choice == '1':
                self.show_settings()
            elif choice == '2':
                 confirm = input(Fore.YELLOW + "آیا از پاک کردن تمام فایل‌های کش مطمئن هستید؟ (y/n): " + Fore.RESET).lower()
                 if confirm == 'y':
                      count = self.model.cache.clear_cache()
                      print(Fore.GREEN + f"{count} فایل کش حذف شد." + Fore.RESET)
                 else:
                      print(Fore.BLUE + "عملیات پاک کردن کش لغو شد." + Fore.RESET)
            elif choice == '3':
                 topic = self.get_valid_input("موضوع مورد نظر برای تحقیق کلمات کلیدی: ", min_length=3)
                 print(Fore.YELLOW + f"\nدر حال اجرای تحقیق کلمات کلیدی (Placeholder) برای '{topic}'..." + Style.RESET_ALL)
                 result = self.model.research_keywords(topic)
                 print(Fore.GREEN + "\nنتیجه (Placeholder):" + Fore.RESET)
                 print(f"وضعیت: {result.get('status')}")
                 print(f"توصیه اولیه:\n{result.get('placeholder_advice')}")
            elif choice == '4':
                 url = self.get_valid_url("آدرس URL رقیب برای تحلیل: ")
                 if url:
                     print(Fore.YELLOW + f"\nدر حال اجرای تحلیل رقیب (Placeholder) برای '{url}'..." + Style.RESET_ALL)
                     result = self.model.analyze_competitor(url)
                     print(Fore.GREEN + "\nنتیجه (Placeholder - خلاصه فنی):" + Fore.RESET)
                     print(f"وضعیت: {result.get('status')}")
                     summary = result.get('technical_audit_summary', {})
                     print(f"  موفقیت درخواست: {'✅' if summary.get('success') else '❌'} {summary.get('error') or ''}")
                     if summary.get('success'):
                          print(f"  عنوان: {summary.get('title')}")
                          print(f"  تعداد H1: {summary.get('h1_count')}")
                          print(f"  زمان بارگذاری: {summary.get('load_time')}")

            elif choice == '5':
                 topic = self.get_valid_input("موضوع مورد نظر برای تحلیل ترندها: ", min_length=3)
                 print(Fore.YELLOW + f"\nدر حال اجرای تحلیل ترندها (Placeholder) برای '{topic}'..." + Style.RESET_ALL)
                 result = self.model.analyze_trends(topic)
                 print(Fore.GREEN + "\nنتیجه (Placeholder):" + Fore.RESET)
                 print(f"وضعیت: {result.get('status')}")
                 print(f"توصیه اولیه:\n{result.get('placeholder_advice')}")

            elif choice == '0':
                break # خروج از منوی تنظیمات
            else:
                print(Fore.RED + "گزینه نامعتبر." + Fore.RESET)

            input(Fore.LIGHTBLACK_EX + "\nEnter برای ادامه در منوی تنظیمات..." + Fore.RESET)


    def show_settings(self):
        """نمایش تنظیمات فعلی"""
        print(Fore.YELLOW + Style.BRIGHT + "\n--- تنظیمات فعلی سیستم ---" + Style.RESET_ALL)
        try:
             # تبدیل dataclass به دیکشنری برای نمایش بهتر
             settings_dict = asdict(self.config)
             # حذف موارد خیلی طولانی یا غیر ضروری برای نمایش
             settings_dict.pop('SEO_KNOWLEDGE_BASE', None)

             for key, value in settings_dict.items():
                  # نمایش مسیرها به صورت کوتاه‌تر
                  if isinstance(value, str) and ('DIR' in key or 'PATH' in key):
                       value_display = f"./{Path(value).relative_to(Path.cwd())}" if Path(value).is_relative_to(Path.cwd()) else value
                  else:
                      value_display = value
                  print(f"- {Fore.CYAN}{key}{Fore.RESET}: {Fore.WHITE}{value_display}{Fore.RESET}")

        except Exception as e:
             print(Fore.RED + f"خطا در نمایش تنظیمات: {e}" + Fore.RESET)


    def exit_app(self):
        """7. خروج از برنامه"""
        self.logger.info("درخواست خروج از برنامه.")
        print(Fore.YELLOW + Style.BRIGHT + "\nبا تشکر از استفاده از مدل متخصص سئو. خدانگهدار!" + Style.RESET_ALL)
        sys.exit(0)


# --- تابع اصلی ---
def main():
    """تابع اصلی اجرای برنامه"""
    global logger # استفاده از logger سراسری

    try:
        print(Fore.CYAN + Style.BRIGHT + "\nدر حال راه‌اندازی مدل متخصص سئو..." + Style.RESET_ALL)

        # دانلود منابع NLTK در ابتدای کار
        download_nltk_resources()

        # Initialize configuration
        config = SEOExpertConfig()

        # Display system info
        print(Fore.CYAN + "\n=== اطلاعات سیستم ===" + Fore.RESET)
        print(f"مدل پایه: {Fore.YELLOW}{config.MODEL_NAME}{Fore.RESET}")
        print(f"دستگاه محاسباتی: {Fore.YELLOW}{config.DEVICE.upper()}{Fore.RESET}")
        if config.DEVICE == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU: {Fore.YELLOW}{gpu_name} ({gpu_memory:.1f} GB){Fore.RESET}")
            except Exception as e:
                print(Fore.YELLOW + "اطلاعات GPU قابل دریافت نیست." + Fore.RESET)
        print(f"مسیر پایه: {Fore.YELLOW}{config.BASE_DIR}{Fore.RESET}")
        print(f"مسیر مدل خروجی: {Fore.YELLOW}{config.OUTPUT_DIR}{Fore.RESET}")
        print(f"مسیر گزارش‌ها: {Fore.YELLOW}{config.REPORTS_DIR}{Fore.RESET}")
        print(f"مسیر کش: {Fore.YELLOW}{config.CACHE_DIR}{Fore.RESET}")
        print(Fore.CYAN + "=====================" + Fore.RESET)


        # Initialize model
        print(Fore.YELLOW + "\nدر حال بارگذاری مدل و توکنایزر (ممکن است کمی طول بکشد)..." + Style.RESET_ALL)
        seo_model = SEOExpertModel(config)
        print(Fore.GREEN + "مدل با موفقیت بارگذاری شد." + Fore.RESET)


        # آموزش مدل (در صورت نیاز)
        if input(Fore.CYAN + "\nآیا می‌خواهید مدل را بر روی داده‌های سفارشی آموزش دهید؟ (y/n): " + Fore.RESET).lower().strip() == 'y':
            print(Fore.YELLOW + "\n--- شروع فرآیند آموزش مدل ---" + Fore.RESET)
            print(Fore.LIGHTRED_EX + "هشدار: آموزش مدل می‌تواند بسیار زمان‌بر باشد و به منابع محاسباتی زیادی (مخصوصاً GPU و RAM) نیاز دارد." + Fore.RESET)
            print(f"داده‌های آموزش از: {config.TRAIN_DATA_PATH}")
            print(f"داده‌های ارزیابی از: {config.TEST_DATA_PATH}")
            print(f"مدل نهایی در: {config.OUTPUT_DIR}")
            if input(Fore.CYAN + "آیا از ادامه آموزش مطمئن هستید؟ (y/n): " + Fore.RESET).lower().strip() == 'y':
                try:
                    start_train_time = time.time()
                    metrics = seo_model.train()
                    end_train_time = time.time()
                    if metrics:
                         print(Fore.GREEN + Style.BRIGHT + f"\nآموزش با موفقیت کامل شد! (زمان: {end_train_time - start_train_time:.2f} ثانیه)" + Style.RESET_ALL)
                         print(f"آخرین Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
                         if 'eval_loss' in metrics: print(f"بهترین Eval Loss: {metrics.get('eval_loss', 'N/A'):.4f}")
                         print(f"نتایج کامل در: {Path(config.OUTPUT_DIR) / 'training_metrics.json'}")
                    else:
                         print(Fore.RED + "فرآیند آموزش با خطا مواجه شد یا داده‌ای یافت نشد." + Fore.RESET)
                except Exception as e:
                    logger.error(f"خطا در فرآیند آموزش آغاز شده توسط کاربر: {e}", exc_info=True)
                    print(Fore.RED + Style.BRIGHT + f"\nخطا در حین آموزش: {str(e)}" + Style.RESET_ALL)
                    # برنامه می‌تواند ادامه دهد یا خارج شود. فعلا ادامه می‌دهیم.
            else:
                print(Fore.BLUE + "آموزش مدل لغو شد." + Fore.RESET)
        else:
            print(Fore.BLUE + "آموزش مدل رد شد. از مدل موجود استفاده می‌شود." + Fore.RESET)


        # راه‌اندازی رابط کاربری
        print(Fore.GREEN + "\nراه‌اندازی رابط کاربری..." + Fore.RESET)
        ui = SEOExpertUI(seo_model)
        ui.run() # شروع حلقه اصلی برنامه

    except RuntimeError as e:
         # خطاهای زمان اجرا که در بارگذاری مدل/توکنایزر رخ داده اند
         logger.critical(f"خطای زمان اجرا در هنگام راه‌اندازی: {e}", exc_info=True)
         print(Fore.RED + Style.BRIGHT + f"\nخطای بحرانی در راه‌اندازی: {e}. برنامه نمی‌تواند ادامه دهد." + Style.RESET_ALL)
         sys.exit(1)
    except FileNotFoundError as e:
         logger.critical(f"فایل ضروری یافت نشد: {e}", exc_info=True)
         print(Fore.RED + Style.BRIGHT + f"\nخطای بحرانی: فایل ضروری یافت نشد: {e}. برنامه خاتمه می‌یابد." + Style.RESET_ALL)
         sys.exit(1)
    except KeyboardInterrupt:
        logger.info("خروج از برنامه توسط کاربر در حین راه‌اندازی.")
        print(Fore.YELLOW + "\nعملیات راه‌اندازی لغو شد. خدانگهدار!" + Style.RESET_ALL)
        sys.exit(0)
    except Exception as e:
        logger.critical(f"خطای غیرمنتظره در تابع main: {e}", exc_info=True)
        print(Fore.RED + Style.BRIGHT + f"\nخطای غیرمنتظره و بحرانی رخ داد: {str(e)}" + Style.RESET_ALL)
        print(Fore.YELLOW + "برای جزئیات بیشتر به فایل seo_expert.log مراجعه کنید." + Fore.RESET)
        sys.exit(1)

if __name__ == "__main__":
    main()
