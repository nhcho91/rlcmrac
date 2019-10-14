# README

## Installation

`environment.yml` 파일을 이용해서 `conda` 가상환경을 만든다.

```bash
$ conda env create -f environment.yml
```

가상환경은 프로젝트 루트 디렉터리에 `.envs` 라는 이름으로 만들어진다.
이제, `.envs`를 활성화 할 수 있다.

```bash
$ conda activate ./.envs
```

긴 prefix를 줄이고 싶다면 다음 명령어를 입력한다
```bash
$ conda config --set env_prompt '({name})'
```

다음으로 `fym` 패키지를 설치한다.

```bash
$ cd ..
$ git clone https://github.com/fdcl-nrf/fym.git
$ cd fym
$ pip install -e .
```
