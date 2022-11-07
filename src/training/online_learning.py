from selenium import webdriver
from selenium.webdriver.common import keys
from selenium.webdriver.common.by import By
import time
import requests
import random
import os
import re
from pathlib import Path
from torch import nn
from PIL import Image
from zhon.hanzi import punctuation
 

class OnlineLearner(nn.Module):

    def __init__(self, savedir, args, max_n=1):
        super().__init__()
        self.markPath=f"mark_log/mark.txt"
        self.unurl = f"mark_log/unurl.txt"
        self.savedir = savedir
        self.max_n = max_n + 1
        self.rank = args.rank
        self.fail_photo = Image.open("assets/fail.jpg") # 访问fail图片路径

        self.options=webdriver.ChromeOptions()
        self.options.add_argument('--headless')
        self.options.add_argument('--no-sandbox')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('start-maximized')
        self.options.add_argument("--disable-extensions")
        self.options.add_argument('--disable-browser-side-navigation')
        self.options.add_argument('enable-automation')
        self.options.add_argument('--disable-infobars')
        self.options.add_argument('enable-features=NetworkServiceInProcess')
        self.options.add_argument("disable-blink-features=AutomationControlled")
        self.browser = webdriver.Chrome(options=self.options)
        time.sleep(2)
        self.browser.get("https://www.taobao.com/")
        time.sleep(2)
        # 找到搜索框输入内容并搜索
        self.browser.find_element(By.XPATH, '//*[@id="q"]').send_keys("便携果汁杯", keys.Keys.ENTER)
        time.sleep(1)

        # 切换成二维码登录
        self.browser.find_element(By.XPATH, '//*[@id="login"]/div[1]/i').click()

        # 判断当前页面是否为登录页面
        while self.browser.current_url.startswith("https://login.taobao.com/"):
            self.browser.save_screenshot(f'barcode_{self.rank}.jpg')
            print(f"请扫描二维码 barcode_{self.rank}.jpg 进行登录")
            time.sleep(5)
    
        print(f"barcode_{self.rank} 登录成功!!!")
        time.sleep(10)

    # 创建浏览器
    def functions(self, raw_text):
        for line in raw_text:
            line = line.strip()
            dir = line.replace(" ", "")
            dir = re.sub('[{}]'.format(punctuation),"", dir)

            try:
                n = 1
                if not os.path.exists(self.savedir + "/{}".format(dir)):
                    os.mkdir(self.savedir + "/{}".format(dir))
                    self.fail_photo.save(self.savedir + f"/{dir}/0_{n}.jpg")
                else:
                    if os.path.exists(self.savedir + f"/{dir}/0_{n}.jpg") or (os.path.exists(self.savedir + f"/{dir}/{n}.jpg") and Path(self.savedir + f"/{dir}/{n}.jpg").stat().st_size >= 500):
                        continue
                    else:
                        self.fail_photo.save(self.savedir + f"/{dir}/0_{n}.jpg")
                
                time.sleep(1)
                self.browser.find_element(By.XPATH, '//*[@id="q"]').clear()
                self.browser.find_element(By.XPATH, '//*[@id="q"]').send_keys(dir, keys.Keys.ENTER)                
                time.sleep(1)
                items = self.browser.find_elements(By.CSS_SELECTOR, '.m-itemlist .items > div')
            except:
                continue

            for idx, item in enumerate(items):
                # 获取这张图片的下载地址
                try:
                    img = item.find_element(By.CSS_SELECTOR, ".pic-box .pic img").get_attribute("data-src")
                except:
                    if idx == len(items):
                        with open(self.markPath, "a") as markfile:
                            markfile.write(f"{dir}: {n}")
                        markfile.close()
                        print("\n{}下载失败,追加至{}。".format(dir, self.markPath))
                        break
                    else:
                        continue
                try:
                    # 拼接完成的下载地址
                    if img[0:4]!='http':
                        img_url = "http:" + img
                    time.sleep(1)
                    # print(img_url)
                    # 通过requests下载这张图片
                    sleep_time = random.random()*10
                    time.sleep(sleep_time)
                    # 文件夹需要手动创建好
                    file = open(self.savedir + f"/{dir}/{n}.jpg", "wb")
                    file.write(requests.get(img_url).content)
                    print("{}: 下载图片".format(dir) + str(n) + ": "+ img_url)
                except:
                    with open(self.unurl, "a") as unurl:
                            unurl.write(f"{dir}: {n}\n")
                    self.fail_photo.save(self.savedir + f"/{dir}/{n}.jpg")
                    continue

                if Path(self.savedir + f"/{dir}/{n}.jpg").stat().st_size < 500:
                    print("{}: 图片".format(dir) + str(n) + "下载失败: "+ img_url)
                    os.remove(self.savedir + f"/{dir}/{n}.jpg")
                    n = n - 1

                    if idx == len(items) - 1:
                        with open(self.markPath, "a") as markfile:
                            markfile.write(f"{dir}: {n}")
                        markfile.close()
                        print("\n{}下载失败,追加至{}。".format(dir, self.markPath))
                        self.fail_photo.save(self.savedir + f"/{dir}/{n+1}.jpg")
                        break
                else:
                    im = Image.open(self.savedir + f"/{dir}/{n}.jpg") # 访问图片路径
                    img = im.resize((224, 224)).convert('RGB')
                    img.save(self.savedir + f"/{dir}/{n}.jpg")
                    if os.path.exists(self.savedir + f"/{dir}/0_{n}.jpg"):
                        os.remove(self.savedir + f"/{dir}/0_{n}.jpg")

                n += 1
                if n == self.max_n:
                    break




 
 