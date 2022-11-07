from pyvirtualdisplay import Display

from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('start-maximized')
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument('--disable-browser-side-navigation')
chrome_options.add_argument('enable-automation')
chrome_options.add_argument('--disable-infobars')
chrome_options.add_argument('enable-features=NetworkServiceInProcess')
display=Display(visible=0, size=(800, 600))# 初始化屏幕 display.start()
driver=webdriver.Chrome(options=chrome_options)# 初始化Chrome

driver.get('http://www.baidu.com/')

print(driver.title)
print("安装成功！")

