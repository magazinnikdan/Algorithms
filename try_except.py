import os

os.chdir('\\Users\\magaz\\Downloads')


def parse_file(filename):
    with open(filename, 'r', encoding='utf-8') as file_in:
        f = file_in.readlines()
        country_info = dict()
        line_count = 1

        for line in f:
            # line_count = = enumerate(f)
            content = line.split(',')


            # for (i, line) in enumerate(f):
            #     line_count = i

            if line == '\n':
                continue
            else:
                line_count += 1

            try:
                info = (float(content[1]), float(content[2]), float(content[3]), int(content[4]))


            except ValueError:
                print(f"Line {line_count} was ignored due to malformed content.")

            except IndexError:
                print(f"Line {line_count} was ignored due to missing information.")
            else:
                country_info[content[0]] = info
        return country_info


print(parse_file("nettle_1999_climate.csv"))