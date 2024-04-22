# Importando as bibliotecas 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# lista de filmes e suas descrições
filmes = [
    {'titulo': 'Matrix', 'descricao': 'Ficção científica: Um hacker descobre a verdade sobre a realidade.'},
    {'titulo': 'Interestelar', 'descricao': 'Ficção científica: Uma equipe de exploradores viaja através de um buraco de minhoca em busca de um novo lar para a humanidade.'},
    {'titulo': 'O Senhor dos Anéis', 'descricao': 'Fantasia: Um hobbit relutante parte em uma jornada para destruir um anel mágico.'},
    {'titulo': 'Pulp Fiction', 'descricao': 'Crime: Histórias interligadas de criminosos em Los Angeles.'},
    {'titulo': 'Clube da Luta', 'descricao': 'Drama: Um homem desiludido forma um clube de luta subterrâneo como forma de terapia.'},
    {'titulo': 'Avatar', 'descricao': 'Ficção científica: Em um planeta alienígena, um ex-fuzileiro naval é recrutado para resgatar uma mulher raptada por alienígenas.'},
    {'titulo': 'Jurassic Park', 'descricao': 'Aventura: Um parque temático de dinossauros tem uma falha de segurança e os dinossauros começam a escapar.'},
    {'titulo': 'Forrest Gump', 'descricao': 'Drama: A vida de Forrest Gump, um homem com QI abaixo da média, que participou de vários eventos históricos.'},
    {'titulo': 'Titanic', 'descricao': 'Romance: Um romance épico sobre o naufrágio do RMS Titanic.'},
    {'titulo': 'De Volta para o Futuro', 'descricao': 'Ficção científica: Um adolescente viaja no tempo para evitar que seus pais se conheçam.'},
    {'titulo': 'O Rei Leão', 'descricao': 'Animação: Um leão jovem e seu destino de se tornar rei após a morte de seu pai.'},
    {'titulo': 'O Poderoso Chefão', 'descricao': 'Crime: A história da família Corleone, liderada por Don Vito Corleone.'},
    {'titulo': 'Batman Begins', 'descricao': 'Ação: A história de origem do super-herói Batman e sua luta contra o crime em Gotham City.'},
    {'titulo': 'Harry Potter e a Pedra Filosofal', 'descricao': 'Fantasia: Um jovem bruxo descobre seu destino e frequenta a Escola de Magia e Bruxaria de Hogwarts.'},
    {'titulo': 'O Iluminado', 'descricao': 'Terror: Um escritor aceita um emprego como zelador de inverno em um hotel isolado, onde eventos assustadores começam a acontecer.'},
    {'titulo': 'O Exterminador do Futuro', 'descricao': 'Ficção científica: Um ciborgue é enviado de volta no tempo para assassinar a mãe de um futuro líder da resistência humana.'},
    {'titulo': 'Os Vingadores', 'descricao': 'Ação: Super-heróis se unem para combater uma ameaça intergaláctica.'},
    {'titulo': 'O Labirinto do Fauno', 'descricao': 'Fantasia: Durante a Guerra Civil Espanhola, uma jovem grávida escapa para um mundo fantástico onde enfrenta uma série de desafios mágicos.'},
    {'titulo': 'O Show de Truman', 'descricao': 'Drama: Um homem comum descobre que sua vida é um programa de televisão de realidade ao vivo.'},
    {'titulo': 'A Origem', 'descricao': 'Ficção científica: Um ladrão habilidoso rouba segredos corporativos através do uso de tecnologia de compartilhamento de sonhos.'},
    {'titulo': 'Os Infiltrados', 'descricao': 'Crime: Um policial disfarçado e um espião da máfia tentam identificar uns aos outros enquanto infiltrados em uma organização criminosa.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'Scarface', 'descricao': 'Crime: Um imigrante cubano sobe para se tornar um poderoso traficante de drogas em Miami.'},
    {'titulo': 'A Lista de Schindler', 'descricao': 'Drama: Um empresário alemão salva a vida de mais de mil refugiados judeus durante o Holocausto.'},
    {'titulo': 'Gladiador', 'descricao': 'Ação: Um general romano é traído e sua família é assassinada. Ele busca vingança como gladiador em uma arena.'},
    {'titulo': 'O Lobo de Wall Street', 'descricao': 'Crime: A história real de um corretor da bolsa de valores que se envolve em corrupção e fraude em Wall Street.'},
    {'titulo': 'Jurassic World', 'descricao': 'Aventura: Um parque temático de dinossauros é reaberto, mas as coisas dão errado quando um novo dinossauro geneticamente modificado escapa.'},
    {'titulo': 'Os Oito Odiados', 'descricao': 'Western: Durante uma tempestade de neve, estranhos se reúnem em uma hospedaria no Wyoming após a Guerra Civil Americana.'},
    {'titulo': 'Cidade de Deus', 'descricao': 'Crime: A história de dois amigos crescendo em uma favela do Rio de Janeiro, um se torna fotógrafo enquanto o outro se torna traficante de drogas.'},
    {'titulo': 'Glória Feita de Sangue', 'descricao': 'Drama: Durante a Primeira Guerra Mundial, soldados franceses enfrentam julgamento por "covardia em face do inimigo" após se recusarem a participar de um ataque suicida.'},
    {'titulo': 'Taxi Driver', 'descricao': 'Drama: Um veterano do Vietnã se torna um taxista insone na cidade de Nova York, onde ele vê a sociedade em sua pior forma e planeja como limpá-la.'},
    {'titulo': 'A Origem', 'descricao': 'Ficção científica: Um ladrão habilidoso rouba segredos corporativos através do uso de tecnologia de compartilhamento de sonhos.'},
    {'titulo': 'Os Bons Companheiros', 'descricao': 'Crime: A história verdadeira de Henry Hill e seus amigos, que trabalharam para a máfia e entraram em conflito com ela em meados do século 20 em Nova York.'},
    {'titulo': 'A Queda: As Últimas Horas de Hitler', 'descricao': 'Histórico: A queda de Adolf Hitler e o Terceiro Reich conforme testemunhada e contada por sua secretária Traudl Junge.'},
    {'titulo': 'Psicose', 'descricao': 'Terror: Uma secretária em fuga rouba dinheiro e se hospeda em um motel remoto, dirigido por um homem perturbado e sua mãe dominadora.'},
    {'titulo': 'Os Suspeitos', 'descricao': 'Crime: Um detetive experiente e seu parceiro novato investigam um seqüestro que se torna um jogo mortal com um serial killer.'},
    {'titulo': 'Os Intocáveis', 'descricao': 'Crime: Durante a década de 1930, o agente do Tesouro dos Estados Unidos Eliot Ness monta uma equipe para derrubar o chefe do crime Al Capone.'},
    {'titulo': 'O Sexto Sentido', 'descricao': 'Terror: Um garoto vê e fala com os mortos e busca a ajuda de um psicólogo infantil para superar seu medo.'},
    {'titulo': 'O Resgate do Soldado Ryan', 'descricao': 'Guerra: Após o desembarque do Dia D, um grupo de soldados americanos é enviado para resgatar um soldado para que sua mãe não o perca para a guerra.'},
    {'titulo': 'A Origem', 'descricao': 'Ficção científica: Um ladrão habilidoso rouba segredos corporativos através do uso de tecnologia de compartilhamento de sonhos.'},
    {'titulo': 'O Artista', 'descricao': 'Drama: Na Hollywood da década de 1920, um astro do cinema mudo enfrenta o surgimento do cinema falado e a perda de sua fama.'},
    {'titulo': 'Whiplash: Em Busca da Perfeição', 'descricao': 'Drama: Um jovem baterista de jazz é abusado mentalmente por um instrutor obcecado pela perfeição.'},
    {'titulo': 'A Vida é Bela', 'descricao': 'Drama: Durante o Holocausto, um homem usa sua imaginação para proteger seu filho dos horrores de um campo de concentração.'},
    {'titulo': 'Um Sonho de Liberdade', 'descricao': 'Drama: Dois homens encarcerados formam uma amizade única e passam décadas planejando sua fuga de uma prisão de segurança máxima.'},
    # Adicione mais filmes conforme necessário
    {'titulo': 'O Grande Gatsby', 'descricao': 'Drama: Um escritor narra sua vida ao conhecer Jay Gatsby, um milionário misterioso, que é obcecado por sua ex-namorada.'},
    {'titulo': 'A Teoria de Tudo', 'descricao': 'Drama: A história do renomado astrofísico Stephen Hawking e seu relacionamento com sua esposa Jane.'},
    {'titulo': 'O Discurso do Rei', 'descricao': 'Drama: A história real do rei George VI do Reino Unido, sua luta contra a gagueira e a relação com seu terapeuta Lionel Logue.'},
    {'titulo': 'Moulin Rouge', 'descricao': 'Musical: Um escritor se apaixona por uma cortesã no famoso cabaré Moulin Rouge em Paris.'},
    {'titulo': 'O Artista', 'descricao': 'Drama: Um astro do cinema mudo enfrenta o surgimento do cinema falado e a perda de sua fama.'},
    {'titulo': 'Amor', 'descricao': 'Drama: A história de um casal idoso enfrentando as dificuldades do envelhecimento e da doença.'},
    {'titulo': 'Menina de Ouro', 'descricao': 'Drama: Uma treinadora de boxe aposentada aceita treinar uma boxeadora talentosa, desafiando as expectativas de gênero.'},
    {'titulo': 'Chicago', 'descricao': 'Musical: Duas assassinas famosas enfrentam o sistema de justiça criminal de Chicago enquanto buscam a fama.'},
    {'titulo': 'Erin Brockovich: Uma Mulher de Talento', 'descricao': 'Drama: A história real de uma mãe solteira que ajuda a expor uma empresa que contaminou o suprimento de água de uma cidade.'},
    {'titulo': 'O Artista', 'descricao': 'Drama: Um astro do cinema mudo enfrenta o surgimento do cinema falado e a perda de sua fama.'},
    {'titulo': 'O Garoto', 'descricao': 'Drama: Um jovem encontra e cuida de um bebê abandonado, formando um vínculo inquebrável.'},
    {'titulo': 'A Chegada', 'descricao': 'Ficção científica: Uma linguista é recrutada para comunicar-se com extraterrestres que chegaram à Terra.'},
    {'titulo': 'Sociedade dos Poetas Mortos', 'descricao': 'Drama: Um professor de literatura inspira seus alunos a desafiar as convenções sociais e a perseguir seus sonhos.'},
    {'titulo': 'La La Land: Cantando Estações', 'descricao': 'Musical: Um pianista de jazz e uma aspirante a atriz se apaixonam enquanto perseguem seus sonhos em Los Angeles.'},
    {'titulo': 'O Show de Truman', 'descricao': 'Drama: Um homem descobre que sua vida é um programa de televisão de realidade ao vivo.'},
    {'titulo': 'Três Anúncios para um Crime', 'descricao': 'Drama: Uma mãe desafia a polícia local a resolver o assassinato de sua filha, fazendo três anúncios controversos.'},
    {'titulo': 'Crepúsculo dos Deuses', 'descricao': 'Drama: Um roteirista falido se envolve com uma atriz decadente, levando a um desfecho trágico.'},
    {'titulo': 'O Artista', 'descricao': 'Drama: Um astro do cinema mudo enfrenta o surgimento do cinema falado e a perda de sua fama.'},
    {'titulo': 'A Origem', 'descricao': 'Ficção científica: Um ladrão habilidoso rouba segredos corporativos através do uso de tecnologia de compartilhamento de sonhos.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'Clube dos Cinco', 'descricao': 'Drama: Cinco adolescentes de diferentes origens se encontram em detenção e aprendem que têm mais em comum do que pensavam.'},
    {'titulo': 'O Silêncio dos Inocentes', 'descricao': 'Thriller: Um jovem agente do FBI busca a ajuda de um psiquiatra canibal para capturar um assassino em série.'},
    {'titulo': 'As Branquelas', 'descricao': 'Comédia: Dois agentes do FBI se disfarçam como mulheres socialites para proteger duas herdeiras de um sequestro.'},
    {'titulo': 'Se Beber, Não Case!', 'descricao': 'Comédia: Três amigos acordam de uma noite de bebedeira em Las Vegas sem lembrar do que aconteceu e tentam encontrar o noivo desaparecido.'},
    {'titulo': 'Todo Mundo em Pânico', 'descricao': 'Comédia, Terror: Uma paródia de filmes de terror populares, onde um grupo de adolescentes enfrenta eventos estranhos.'},
    {'titulo': 'Superbad - É Hoje', 'descricao': 'Comédia: Dois amigos desajeitados tentam comprar bebidas para uma festa e acabam em uma série de situações hilárias.'},
    {'titulo': 'Anjos da Lei', 'descricao': 'Comédia, Ação: Dois policiais disfarçados voltam à escola para desmantelar uma operação de tráfico de drogas.'},
    {'titulo': 'Loucademia de Polícia', 'descricao': 'Comédia: Um grupo de recrutas desajustados entra para a academia de polícia e enfrenta um instrutor rigoroso.'},
    {'titulo': 'Quem Vai Ficar com Mary?', 'descricao': 'Comédia, Romance: Um homem se apaixona pela mesma mulher que seu amigo de infância e tenta conquistá-la.'}
   
]

# DataFrame do pandas com os dados dos filmes
df = pd.DataFrame(filmes)

# Criando um vetorizador TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Construindo a matriz TF-IDF
matriz_tfidf = tfidf.fit_transform(df['descricao'])

#  algoritmo K-means para agrupar os filmes em 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(matriz_tfidf)

# Colocando os clusters no DataFrame
df['cluster'] = clusters

# Função para obter recomendações de filmes com base no cluster
def recomendar_filmes_por_genero(genero, numero_recomendacoes=3):
    filmes_genero = df[df['descricao'].str.contains(genero, case=False)]
    if len(filmes_genero) > 0:
        cluster_id = filmes_genero['cluster'].iloc[0]  # Pegar o cluster do primeiro filme do gênero
        filmes_cluster = df[df['cluster'] == cluster_id]
        return filmes_cluster.sample(min(numero_recomendacoes, len(filmes_cluster)))['titulo'].tolist()
    else:
        return []

# Solicitar ao usuário que insira um gênero de filme para receber recomendações
genero_filme_usuario = input("Digite um gênero de filme que você gosta para receber recomendações: ")

# Puxando recomendações com base no gênero de filme inserido pelo usuário
recomendacoes = recomendar_filmes_por_genero(genero_filme_usuario)
if recomendacoes:
    print(f"Recomendações de filmes similares ao gênero '{genero_filme_usuario}':")
    for filme in recomendacoes:
        print(filme)
else:
    print("Não foram encontradas recomendações para o gênero de filme inserido.")
