#face recognation 

`docker-compose -f prod.yml up -d --build`
`docker-compose -f prod.yml logs app`

# Request in body

`base_url/compare-faces/`

`{
    "url1":"https://api-digital.tsul.uz/storage/user_images_new/50202035910012.png",
    "url2":"http://127.0.0.1:8000/media/users/Safeimagekit-crop-image-to-100x100-maskasiz_7gnXisd.jpg"
}`