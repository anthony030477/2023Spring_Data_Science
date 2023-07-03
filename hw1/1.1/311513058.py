from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import json
from queue import Queue
from threading import Thread, Lock

import click

@click.group()
def cli():
    pass

@cli.command()
def crawl():
    fist_index=3500
    l=0
    for i in range(3500,3700):
        payload={'from':'/bbs/Beauty/M.1514740613.A.FF1.html','yes':'yes'}
        rs=requests.session()
        res=rs.post("https://www.ptt.cc/ask/over18",data=payload)

        url="https://www.ptt.cc/bbs/Beauty/index"+str(i)+".html"
        res=rs.get(url)
        content=res.text
        if l>0:
            break
        soup=BeautifulSoup(content,'html.parser')#標籤樹
        rent_list= soup.find_all(class_="r-ent")
        for it in rent_list:
            link=it.find(class_='title')
            if link.find_all('a'):
                date = it.find_all(class_="date")[0].string
                if date==' 1/01':
                    fist_index=i
                    l=l+1
                    break

    dict_list=[]
    dict_list_popular=[]
    for a in tqdm(range(310)):
        payload={'from':'/bbs/Beauty/M.1514740613.A.FF1.html','yes':'yes'}
        rs=requests.session()
        res=rs.post("https://www.ptt.cc/ask/over18",data=payload)

        url="https://www.ptt.cc/bbs/Beauty/index"+str(fist_index+a)+".html"
        
            
        #time.sleep(0.5)
        res=rs.get(url)
        content=res.text

        soup=BeautifulSoup(content,'html.parser')#標籤樹
        rent_list= soup.find_all(class_="r-ent")
        for it in rent_list:
            link=it.find(class_='title')
            if link.find_all('a'):
                dict={}
                link=link.find_all('a')
                title=link[0].string
                if '[公告]' in title:
                    continue
                dict['title']=title
                url_link=link[0].get('href')
                if url_link=='/bbs/Beauty/M.1672503968.A.5B5.html':
                    break
                dict['url']="https://www.ptt.cc"+url_link
                date = it.find_all(class_="date")[0].string
                if a< 10 and (date=='12/31' or date=='12/30') :
                    continue
                if a> 250 and (date==' 1/01') :
                    break
                if date[0]==' ':
                    date=str(0)+date[1]+date[-2]+date[-1]
                else:
                    date=date[0]+date[1]+date[-2]+date[-1]
                dict['date']=date
                
                dict_list.append(dict)
                dict_popular={}
                if it.find_all(class_="hl f1"):
                    push=it.find_all(class_="hl f1")[0].string
                    if push=='爆':
                        dict_popular['title']=title
                        dict_popular['url']="https://www.ptt.cc"+url_link
                        dict_popular['date']=date
                        dict_list_popular.append(dict)

    with open('all_article.jsonl', 'w',encoding='utf-8') as f:
        for item in dict_list:
            json.dump(item, f,ensure_ascii=False)
            f.write('\n')
    with open('all_popular.jsonl', 'w',encoding='utf-8') as f:
        for item in dict_list_popular:
            json.dump(item, f,ensure_ascii=False)
            f.write('\n')   



def parall(progress,lock,queue,user_dict_list,user_id_list):
    while not queue.empty():
        progress.update()
        url=queue.get()
        payload={'from':'/bbs/Beauty/M.1514740613.A.FF1.html','yes':'yes'}
        rs=requests.session()
        res=rs.post("https://www.ptt.cc/ask/over18",data=payload)
        res=rs.get(url)
        content=res.text

        soup=BeautifulSoup(content,'html.parser')#標籤樹
        push_list= soup.find_all(class_="push")

        for it in push_list:
            user_dict={
                "user_id":None,
                "like_count":None,
                "boo_count":None,
            }
            userid= it.find_all(class_="f3 hl push-userid")
            if userid[0].string in user_id_list:
                for u in user_dict_list:
                    if u['user_id']==userid[0].string:
                        if u["like_count"]==None:
                            user_dict["like_count"]=0

                        if it.find_all(class_="hl push-tag"):
                            u["like_count"]=u["like_count"]+1

                        if u["boo_count"]==None:
                            u["boo_count"]=0

                        if it.find_all(class_="f1 hl push-tag"):
                            booarrowtag_list= it.find_all(class_="f1 hl push-tag")
                            
                            if booarrowtag_list[0].string=='噓 ':
                                u["boo_count"]=u["boo_count"]+1
                        
            else:
                user_dict["user_id"]=userid[0].string
                lock.acquire()
                user_id_list.append(userid[0].string)
                lock.release()
                if user_dict["like_count"]==None:
                    user_dict["like_count"]=0

                if it.find_all(class_="hl push-tag"):
                    user_dict["like_count"]=user_dict["like_count"]+1

                if user_dict["boo_count"]==None:
                    user_dict["boo_count"]=0

                if it.find_all(class_="f1 hl push-tag"):
                    booarrowtag_list= it.find_all(class_="f1 hl push-tag")
                    
                    if booarrowtag_list[0].string=='噓 ':
                        user_dict["boo_count"]=user_dict["boo_count"]+1
                lock.acquire()
                user_dict_list.append(user_dict)
                lock.release()





@cli.command()
@click.argument('start_date')
@click.argument('end_date')
def push(start_date, end_date):
    dict_list_start_end=[]
    with open('all_article.jsonl', 'r',encoding='utf-8') as f:
        
        for line in f:
            # 解析json
            data = json.loads(line)
            # 檢查日期是否符合範圍
            if data['date'] >= start_date and data['date'] <= end_date:
                # 符合條件的資料
                dict_list_start_end.append(data)
    
    
    user_dict_list=[]
    user_id_list=[]

    queue=Queue()#隊列 
    lock = Lock()#防止threads同時對物件做事情
    threads=[]#我的執行序
    
    for a in dict_list_start_end:
       
        url=a['url']
        queue.put(url)#把每個網址都放進去隊列
    progress=tqdm(total=queue.qsize())
    for i in range(20):
        #將你要讓子執行序工作的函式放進去
        threads.append(Thread(target = parall, args = (progress,lock,queue,user_dict_list,user_id_list)))
        
        threads[i].start()#開始工作

    for thread in threads:
        thread.join()#等
        
    
    all_like=0
    all_boo=0
    for u in user_dict_list:
        all_like=all_like+u['like_count']
        all_boo=all_boo+u['boo_count']
    
    like_sorted_user_dict_list = sorted(user_dict_list, key=lambda k: k['like_count'], reverse=True)
    like_sorted_user_dict_list = [d.copy() for d in like_sorted_user_dict_list]
    for d in like_sorted_user_dict_list:
        d['count'] = d.pop('like_count')
        del d['boo_count']
    boo_sorted_user_dict_list=sorted(user_dict_list, key=lambda k: k['boo_count'], reverse=True)
    boo_sorted_user_dict_list = [d.copy() for d in boo_sorted_user_dict_list]
    for d in boo_sorted_user_dict_list:
        d['count'] = d.pop('boo_count')
        del d['like_count']
    like_sorted_user_dict_list=like_sorted_user_dict_list[:10]
    boo_sorted_user_dict_list=boo_sorted_user_dict_list[:10]

    like_sorted_user_dict_list = sorted(
    like_sorted_user_dict_list,
    key=lambda k: (k['count'], k['user_id'],),reverse=True
    )
    boo_sorted_user_dict_list = sorted(
    boo_sorted_user_dict_list,
    key=lambda k: (k['count'], k['user_id']),reverse=True
    )
    push_output={
    "all_like":all_like,
    "all_boo":all_boo,
    }
    for i in range(10):
        push_output["like "+str(i+1)]=like_sorted_user_dict_list[i]
    for i in range(10):
        push_output["boo "+str(i+1)]=boo_sorted_user_dict_list[i]

    filename = f"push_{start_date}_{end_date}.json"
    with open(filename, "w",encoding='utf-8') as f:
        json.dump(push_output, f,indent=2)

@cli.command()
@click.argument('start_date')
@click.argument('end_date')
def popular(start_date, end_date):
    list_popular=[]
    with open('all_popular.jsonl', 'r',encoding='utf-8') as f:
        
        for line in f:
            # 解析json
            data = json.loads(line)
            # 檢查日期是否符合範圍
            if data['date'] >= start_date and data['date'] <= end_date:
                # 符合條件的資料
                list_popular.append(data)
    

    im_url_bao_list=[]
    for i in tqdm(list_popular):
        payload={'from':'/bbs/Beauty/M.1514740613.A.FF1.html','yes':'yes'}
        rs=requests.session()
        res=rs.post("https://www.ptt.cc/ask/over18",data=payload)

        url=i['url']
            
        #time.sleep(0.5)
        res=rs.get(url)
        content=res.text

        soup=BeautifulSoup(content,'html.parser')#標籤樹
        image_list= soup.find_all('a')
        
        for it in image_list:
            if it.get('href') and 'https' in it.get('href') and ('jpg' in it.get('href') or  'jpeg' in it.get('href') or 'png' in it.get('href') or 'gif' in it.get('href')):
                #print(it.get('href'))
                im_url_bao_list.append(it.get('href'))
    
    number_of_popular_articles=len(list_popular)
    popular_output={
    "number_of_popular_articles":number_of_popular_articles,
    "image_urls":im_url_bao_list
    }
    filename = f"popular_{start_date}_{end_date}.json"
    with open(filename, "w",encoding='utf-8') as f:
        json.dump(popular_output, f,indent=2)
 

def parall_keyword(keyword,progress,lock,queue,im_url_keyword_list):
    while not queue.empty():
        progress.update()
        url=queue.get()

        payload={'from':'/bbs/Beauty/M.1514740613.A.FF1.html','yes':'yes'}
        rs=requests.session()
        res=rs.post("https://www.ptt.cc/ask/over18",data=payload)
        res=rs.get(url)
        content=res.text

        soup=BeautifulSoup(content,'html.parser')#標籤樹

        
        main_content_list = soup.find_all(class_='bbs-screen bbs-content')
        text = main_content_list[0].text
        text = text.split('※ 發信站')[0]
        
        if keyword in text:
            image_list= soup.find_all('a')
            for it in image_list:
                if it.get('href') and 'https' in it.get('href') and ('jpg' in it.get('href') or  'jpeg' in it.get('href') or 'png' in it.get('href') or 'gif' in it.get('href')):
                    lock.acquire()
                    im_url_keyword_list.append(it.get('href'))
                    lock.release()
        else:
            continue
        




@cli.command()
@click.argument('keyword')
@click.argument('start_date')
@click.argument('end_date')
def keyword(keyword,start_date, end_date):
    article_list_keyword=[]
    with open('all_article.jsonl', 'r',encoding='utf-8') as f:
        
        for line in f:
            # 解析json
            data = json.loads(line)
            # 檢查日期是否符合範圍
            if data['date'] >= start_date and data['date'] <= end_date:
                # 符合條件的資料
                article_list_keyword.append(data)
    

    queue=Queue()#隊列 
    lock = Lock()#防止threads同時對物件做事情
    threads=[]#我的執行序
    im_url_keyword_list=[]

    for i in article_list_keyword:
        url=i['url']
        queue.put(url)

    progress=tqdm(total=queue.qsize())
    for i in range(20):
        #將你要讓子執行序工作的函式放進去
        threads.append(Thread(target = parall_keyword, args = (keyword,progress,lock,queue,im_url_keyword_list)))
        
        threads[i].start()#開始工作

    for thread in threads:
        thread.join()#等
    
    
    
    keyword_output={
    "image_urls":im_url_keyword_list
    }
    filename = f"keyword_{keyword}_{start_date}_{end_date}.json"
    with open(filename, "w",encoding='utf-8') as f:
        json.dump(keyword_output, f,indent=2)



if __name__ == '__main__':
    cli()

