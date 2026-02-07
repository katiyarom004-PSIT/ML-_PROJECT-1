from setuptools import find_packages,setup
from typing import List


HYPEN_E_DOT='-e .'
def get_requirements(file_packages:str)->List[str]:
    '''
    this function will return the list of get_requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","")for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements        

setup(
    name='ML PROJECT',
    version='3.10.0',
    author='OM',
    author_email='katiyarom004@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')

)