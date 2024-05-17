# SpuriousFL

Personal utils and guides repo

### Stop asking password ssh

In ```localhost``` run:
```
ssh-copy-id -i [path/to/]id_rsa.pub [hostname saved in .ssh/config]
```

## Conda environments chatsheet

### Export conda env

```conda env export > env.yml```


### Install conda env from file

```conda env create -f env.yml```

### Create conda env from scratch

```conda create --name [env_name] python=3.10```

### Delete conda env

```conda env remove --name [env_name]```

### Clone conda env

```conda create --clone [old_env] --name [new_env]```

### List packages env

```conda list```

### List existing conda envs

```conda info --envs```
