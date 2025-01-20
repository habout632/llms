import asyncio
import requests
from concurrent.futures import ThreadPoolExecutor


async def fetch_url(url):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(pool, requests.get, url)
        first_char = response.text[0]
        print(f"first char:{first_char}")
        return first_char


async def main():
    url = "www.baidu.com"
    tasks = []
    for _ in range(4):
        task = asyncio.create_task(fetch_url(url))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    print(f"results: {results}")


if __name__ == '__main__':
    asyncio.run(main())