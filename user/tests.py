from django.test import TestCase
import oss2

# 填写您的Access Key ID和Access Key Secret
access_key_id = ''
access_key_secret = ''

# 填写您的Bucket所在地域
endpoint = 'https://oss-cn-beijing.aliyuncs.com'

# 填写Bucket名称
bucket_name = 'old-care-bucket'

# 创建Bucket对象
bucket = oss2.Bucket(oss2.Auth(access_key_id, access_key_secret), endpoint, bucket_name)
# 填写本地图片文件路径
local_file_path = 'C:\\Users\刘荧\Pictures\\6e557b28f4b898f1c0f26c1e5b5056b.jpg'

# 填写上传到OSS后的文件名
oss_file_name = 'old-care/volunteer/image.jpg'

# 上传文件
bucket.put_object_from_file(oss_file_name, local_file_path)

# 获取文件的URL
image_url = f'https://{bucket_name}.{endpoint.split("//")[1]}/{oss_file_name}'
url = bucket.sign_url('GET', 'old-care/volunteer/image.jpg', 10 * 60)
print(url)
print(f'Image URL: {image_url}')
