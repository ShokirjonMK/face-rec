import asyncio
import aiohttp
import time

async def send_request(session, url, payload):
    try:
        async with session.post(url, json=payload) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        return {"error": str(e)}

async def main():
    start_time = time.time()
    
    url = "http://127.0.0.1:8555/compare-faces/"
    payload = {
        "url1": "https://api-digital.tsul.uz/storage/user_images_new/43110976520017.png",
        "url2": "https://api-digital.tsul.uz/storage/user_images_new/40102941670048.png"
    }
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(50)
        tasks = []
        for _ in range(500):
            async def bounded_request(sem):
                async with sem:
                    return await send_request(session, url, payload)
            tasks.append(bounded_request(semaphore))
        
        responses = await asyncio.gather(*tasks)
        
        success_count = sum(1 for response in responses if "error" not in response)
    
    elapsed_time = time.time() - start_time
    print(f"Successful responses: {success_count} out of 500")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

asyncio.run(main())

